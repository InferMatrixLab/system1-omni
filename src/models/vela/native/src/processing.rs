//! The processor: one admitted request body → predict mapping, plan,
//! and rendered rows, mirroring the reference path
//! (`system_one_to_predict` → `_to_generic` → `_predict`'s row split →
//! `_prep`/`_render`, plus the long-target window rows). Everything the
//! frozen `native-plan-*` fixtures record.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{Map, Value};

use crate::contract::RequestError;
use crate::questions::{PredictQuestion, system_one_to_predict};
use crate::render::{
    Encoder, GenericQuestion, HALU_LABEL, PreparedPart, RMAX, RenderContext, RenderedRow,
    window_part,
};
use crate::state::system_one_state;

pub struct Processor {
    encoder: Encoder,
    max_length: usize,
    router_label_sets: Vec<HashSet<String>>,
    has_broad: bool,
    halu_desc: String,
}

pub struct Planned {
    pub predict_request: Value,
    pub plan: Value,
    pub rows: Vec<RenderedRow>,
    pub window_rows: Vec<RenderedRow>,
}

impl Processor {
    /// Load from an exported native bundle (tokenizer + calibration;
    /// the manifest gate runs first).
    pub fn load(bundle: &Path) -> Result<Self> {
        let manifest = crate::bundle::Bundle::load(bundle)?;
        let calibration: Value = serde_json::from_slice(
            &std::fs::read(bundle.join("calibration.json")).context("calibration.json")?,
        )
        .context("calibration.json must be JSON")?;
        let broad = calibration.get("broad_head").filter(|b| !b.is_null());
        let router_label_sets: Vec<HashSet<String>> = broad
            .and_then(|b| b.get("router_label_sets"))
            .and_then(Value::as_array)
            .filter(|sets| !sets.is_empty())
            .map(|sets| {
                sets.iter()
                    .filter_map(Value::as_array)
                    .map(|labels| {
                        labels
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect()
                    })
                    .collect()
            })
            .unwrap_or_else(|| {
                vec![
                    calibration["pii_types"].as_array().map(|types| {
                        types
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect()
                    }),
                    calibration["halu_schema"]["labels"]
                        .as_object()
                        .map(|labels| labels.keys().cloned().collect()),
                ]
                .into_iter()
                .flatten()
                .collect()
            });
        let has_broad = broad.is_some(); // the bundle gate pinned the broad tensors
        let halu_desc = calibration["halu_schema"]["labels"][HALU_LABEL]
            .as_str()
            .unwrap_or("a claim not supported by the context")
            .to_owned();
        Ok(Self {
            encoder: Encoder::load(&bundle.join("tokenizer.json"))?,
            max_length: manifest.max_length(),
            router_label_sets,
            has_broad,
            halu_desc,
        })
    }

    /// The full planning path for one admitted request body.
    pub fn plan(&self, body: &Map<String, Value>) -> Result<Planned, RequestError> {
        let state = system_one_state(&body["state"])?;
        let questions = body["questions"]
            .as_object()
            .ok_or_else(crate::questions::empty_questions_error)?;
        let mapped = system_one_to_predict(state, questions)?;
        let predict_request = serde_json::to_value(&mapped.request).expect("serializable");
        let plan = serde_json::to_value(&mapped.plan).expect("serializable");
        let state_parts = mapped.request.parts.clone();

        // `_to_generic`: predict questions → generic render questions
        // (Noul is already folded; choice always carries [ABS]).
        let generic: Vec<GenericQuestion> =
            mapped.request.questions.iter().map(to_generic).collect();
        // `_predict`'s row split: non-span questions plus the first span
        // in one row, every further span its own row.
        let spans: Vec<&GenericQuestion> = generic.iter().filter(|q| q.kind == "span").collect();
        let rest: Vec<GenericQuestion> = generic
            .iter()
            .filter(|q| q.kind != "span")
            .cloned()
            .collect();
        let mut rows: Vec<Vec<GenericQuestion>> = Vec::new();
        if !rest.is_empty() || !spans.is_empty() {
            let mut first = rest.clone();
            if let Some(span) = spans.first() {
                first.push((*span).clone());
            }
            rows.push(first);
            for span in spans.iter().skip(1) {
                rows.push(vec![(*span).clone()]);
            }
        }
        let context = RenderContext {
            encoder: &self.encoder,
            max_length: self.max_length,
            router_label_sets: &self.router_label_sets,
            has_broad: self.has_broad,
            halu_desc: self.halu_desc.clone(),
        };
        let span_roles: HashSet<String> =
            spans.iter().flat_map(|q| q.over.iter().cloned()).collect();
        let mut rendered = Vec::new();
        let mut window_rows = Vec::new();
        let mut parts_by_role = std::collections::HashMap::new();
        for part in self.prepared_parts(&state_parts, &span_roles)? {
            parts_by_role.insert(part.role.clone(), part);
        }
        for row in rows {
            if row.is_empty() {
                continue;
            }
            let (adjusted, dispatch) = context.prepare_questions(&row)?;
            let parts: Vec<PreparedPart> = state_parts
                .iter()
                .map(|part| {
                    parts_by_role
                        .get(&part.r#type)
                        .cloned()
                        .expect("prepared part")
                })
                .collect();
            let record = context.render(&parts, &adjusted, &dispatch)?;
            // `_window_rows`: a span target beyond RMAX becomes one row
            // per window (the full-render span block stays on the row).
            let span = row.iter().find(|q| q.kind == "span");
            if let Some(span) = span {
                let index = parts.iter().position(|p| p.role == span.over[0]);
                if let Some(index) = index {
                    let part = &parts[index];
                    let has_words = part.words.is_some();
                    if part.ids.len() > RMAX && has_words {
                        let (_, first_tokens) = part.words.as_ref().expect("words");
                        for (a, b) in crate::render::span_windows(first_tokens, part.ids.len()) {
                            let mut windowed_parts = parts.clone();
                            windowed_parts[index] = window_part(part, a, b);
                            let (w_adjusted, w_dispatch) = context.prepare_questions(&row)?;
                            let mut record =
                                context.render(&windowed_parts, &w_adjusted, &w_dispatch)?;
                            // The runtime's window rows carry the head only.
                            record.window = true;
                            window_rows.push(record);
                        }
                    }
                }
            }
            rendered.push(record);
        }
        Ok(Planned {
            predict_request,
            plan,
            rows: rendered,
            window_rows,
        })
    }

    /// The prepared parts for one request: tokens everywhere, offsets
    /// and word units on span-target parts.
    fn prepared_parts(
        &self,
        state_parts: &[crate::state::Part],
        span_roles: &HashSet<String>,
    ) -> Result<Vec<PreparedPart>, RequestError> {
        state_parts
            .iter()
            .map(|part| {
                let (ids, offsets) = self
                    .encoder
                    .ids_offsets(&part.text)
                    .map_err(|_| RequestError::internal())?;
                if span_roles.contains(&part.r#type) {
                    let words = crate::words::word_first_tokens(&part.text, &offsets);
                    Ok(PreparedPart {
                        role: part.r#type.clone(),
                        ids,
                        offsets: Some(offsets),
                        words: Some(words),
                    })
                } else {
                    Ok(PreparedPart {
                        role: part.r#type.clone(),
                        ids,
                        offsets: None,
                        words: None,
                    })
                }
            })
            .collect()
    }
}

/// `_to_generic` for the SystemOne path: options/labels carry over in
/// order (score levels have empty descriptions), the target part is the
/// resolved `over`, choice always carries `[ABS]` (the reference's
/// `abstain` default is true and the mapper never sets it), and span
/// keeps its threshold and head override.
fn to_generic(question: &PredictQuestion) -> GenericQuestion {
    let over: Vec<String> = match &question.target_part {
        Value::String(role) => vec![role.clone()],
        Value::Array(roles) => roles
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    };
    let (names, descriptions) = match (&question.labels, &question.options) {
        (Some(labels), _) => (
            labels.iter().map(|n| n.name.clone()).collect(),
            labels.iter().map(|n| n.description.clone()).collect(),
        ),
        (None, Some(Value::Array(items))) => {
            // Objects carry name/description; score levels are strings.
            let mut names = Vec::new();
            let mut descriptions = Vec::new();
            for item in items {
                match item {
                    Value::String(name) => {
                        names.push(name.clone());
                        descriptions.push(String::new());
                    }
                    Value::Object(named) => {
                        names.push(named["name"].as_str().unwrap_or_default().to_owned());
                        descriptions
                            .push(named["description"].as_str().unwrap_or_default().to_owned());
                    }
                    _ => {}
                }
            }
            (names, descriptions)
        }
        _ => (Vec::new(), Vec::new()),
    };
    GenericQuestion {
        id: question.id.clone(),
        kind: question.kind.clone(),
        over,
        text: question.text.clone(),
        names,
        descriptions,
        threshold: question.threshold,
        head_override: question.head.clone(),
    }
}
