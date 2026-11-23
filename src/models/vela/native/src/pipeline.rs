//! Executes the processor's complete plan through CUDA readouts and the native heads.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use crate::executor::Executor;
use crate::processing::{Planned, SpanPlan};
use crate::render::{Block, RenderedRow};
use crate::scoring::{
    Calibration, QBlock, SpanOut, aggregate_windows, row_answers, sigmoid, system_one_response,
};

/// Raw head outputs in rendered-row order. Question logits precede set bias.
#[derive(serde::Serialize)]
pub struct RowLogits {
    pub questions: Vec<Vec<f32>>,
    pub span: Option<Vec<Vec<f32>>>,
}

impl Executor {
    /// Execute one request. The observer receives its actual raw head outputs,
    /// indexed as ordinary rows followed by window rows.
    pub fn execute(
        &mut self,
        model_name: &str,
        planned: &Planned,
        calibration: &Calibration,
        mut observe: impl FnMut(usize, &RowLogits),
    ) -> Result<Value> {
        let predict = &planned.predict_request;
        let thresholds = predict["questions"]
            .as_array()
            .context("predict questions")?
            .iter()
            .filter_map(|q| Some((q["id"].as_str()?.to_owned(), q["threshold"].as_f64()?)))
            .collect();
        let part_texts: HashMap<_, _> = predict["parts"]
            .as_array()
            .context("predict parts")?
            .iter()
            .map(|p| {
                Ok((
                    p["type"].as_str().context("part type")?.to_owned(),
                    p["text"].as_str().context("part text")?.to_owned(),
                ))
            })
            .collect::<Result<_>>()?;
        let plan = planned.plan.as_array().context("response plan")?;
        ensure!(
            planned.rows.len() == planned.spans.len(),
            "span plan row count"
        );
        let mut next_window = 0;
        let mut answers = Vec::new();
        let mut span_probs = HashMap::new();
        for (index, (row, span_plan)) in planned.rows.iter().zip(&planned.spans).enumerate() {
            let mut raw = self.score_row(
                row,
                span_plan
                    .as_ref()
                    .is_some_and(|span| !span.windows.is_empty()),
            )?;
            observe(index, &raw);
            let blocks = row.blocks.iter().filter_map(|block| {
                let Block::Question {
                    qid,
                    qtype,
                    names,
                    has_abs,
                    ..
                } = block
                else {
                    return None;
                };
                Some((qid, qtype, names, has_abs))
            });
            let mut questions = Vec::new();
            for ((qid, qtype, names, has_abs), mut logits) in blocks.zip(raw.questions) {
                if qtype == "set" {
                    for logit in &mut logits {
                        *logit += self.heads.set_bias;
                    }
                    ensure!(logits.iter().all(|v| v.is_finite()), "non-finite set logit");
                }
                questions.push(QBlock {
                    qid: qid.clone(),
                    qtype: qtype.clone(),
                    names: names.clone(),
                    has_abs: *has_abs,
                    logits,
                });
            }
            let span = if let Some(span) = span_plan {
                ensure!(span.windows.start == next_window, "window row attribution");
                let output = self.finish_span(row, span, raw.span.take(), planned, &mut observe)?;
                next_window = span.windows.end;
                let entry = plan
                    .iter()
                    .find(|entry| entry["qid"] == span.qid)
                    .context("span response plan")?;
                span_probs.insert(
                    span.qid.clone(),
                    span_probabilities(&output, entry, calibration),
                );
                Some(output)
            } else {
                ensure!(raw.span.is_none(), "span logits have no owner");
                None
            };
            let text = span
                .as_ref()
                .and_then(|span| part_texts.get(&span.over))
                .map(String::as_str);
            answers.extend(row_answers(
                calibration,
                &questions,
                span.as_ref(),
                &thresholds,
                text,
            ));
        }
        ensure!(
            next_window == planned.window_rows.len(),
            "unconsumed window row"
        );
        let input_tokens = planned
            .rows
            .iter()
            .chain(&planned.window_rows)
            .map(|row| row.n_tokens)
            .sum();
        system_one_response(
            model_name,
            plan,
            &answers,
            &span_probs,
            &part_texts,
            input_tokens,
        )
    }

    fn score_row(&mut self, row: &RenderedRow, skip_span: bool) -> Result<RowLogits> {
        // A windowed ordinary row still carries its span block for usage and
        // admission, while only the window rows run that span head.
        let without_span;
        let row = if skip_span {
            without_span = {
                let mut row = row.clone();
                row.blocks
                    .retain(|block| matches!(block, Block::Question { .. }));
                row
            };
            &without_span
        } else {
            row
        };
        let readouts = self.row_readouts(row)?;
        let hidden = self.heads.hidden();
        let valid_hidden =
            |values: &Vec<f32>| values.len() == hidden && values.iter().all(|v| v.is_finite());
        let mut questions = Vec::new();
        let mut readout_questions = readouts.questions.iter();
        for (index, block) in row.blocks.iter().enumerate() {
            let Block::Question {
                qid,
                names,
                ends,
                has_abs,
                ..
            } = block
            else {
                continue;
            };
            let readout = readout_questions
                .next()
                .context("missing question readout")?;
            let count = names.len() + usize::from(*has_abs);
            ensure!(
                readout.block_index == index
                    && &readout.qid == qid
                    && ends.len() == count
                    && readout.candidates.len() == count
                    && readout.candidates.iter().all(valid_hidden),
                "question {qid} readout shape, finiteness or attribution"
            );
            let query = readout
                .query
                .as_ref()
                .filter(|q| valid_hidden(q))
                .context("invalid question query")?;
            let logits = self.heads.forward_candidate(&readout.candidates, query);
            ensure!(
                logits.len() == count && logits.iter().all(|v| v.is_finite()),
                "invalid question {qid} logits"
            );
            questions.push(logits);
        }
        ensure!(
            readout_questions.next().is_none(),
            "unconsumed question readout"
        );
        let span = match readouts.span {
            Some(readout) => {
                let labels = row.names.as_ref().context("span label names")?.len();
                let words = row.w_offs.as_ref().context("span word offsets")?.len();
                ensure!(
                    labels > 0
                        && readout.labels.len() == labels
                        && readout.words.len() == words
                        && readout
                            .labels
                            .iter()
                            .chain(&readout.words)
                            .all(valid_hidden)
                        && readout.query.as_ref().is_some_and(valid_hidden),
                    "invalid span readout"
                );
                let head = match row.head.as_str() {
                    "broad" => &self.heads.broad,
                    "router" => &self.heads.router,
                    other => anyhow::bail!("unknown span head {other}"),
                };
                let logits = head.forward(&readout.words, &readout.labels);
                ensure!(
                    logits.len() == words
                        && logits
                            .iter()
                            .all(|r| r.len() == labels && r.iter().all(|v| v.is_finite())),
                    "invalid span logits"
                );
                Some(logits)
            }
            None => {
                ensure!(
                    !row.blocks
                        .iter()
                        .any(|block| matches!(block, Block::Span { .. })),
                    "missing span readout"
                );
                None
            }
        };
        Ok(RowLogits { questions, span })
    }

    fn finish_span(
        &mut self,
        row: &RenderedRow,
        plan: &SpanPlan,
        ordinary: Option<Vec<Vec<f32>>>,
        planned: &Planned,
        observe: &mut impl FnMut(usize, &RowLogits),
    ) -> Result<SpanOut> {
        let names = row.names.as_ref().context("ordinary span labels")?;
        let logits = if plan.windows.is_empty() {
            ordinary.context("ordinary span logits")?
        } else {
            ensure!(
                ordinary.is_none(),
                "windowed span also ran on its ordinary row"
            );
            let mut logits = Vec::new();
            let mut offsets = Vec::new();
            for index in plan.windows.clone() {
                let window = planned
                    .window_rows
                    .get(index)
                    .context("missing window row")?;
                ensure!(
                    window.window
                        && window.head == row.head
                        && window.names == row.names
                        && window.span_over == row.span_over,
                    "window span attribution"
                );
                let raw = self.score_row(window, false)?;
                ensure!(
                    raw.questions.is_empty(),
                    "window contains non-span question scores"
                );
                observe(planned.rows.len() + index, &raw);
                logits.push(raw.span.context("window span logits")?);
                offsets.push(window.w_offs.clone().context("window offsets")?);
            }
            aggregate_windows(names, &plan.offsets, &logits, &offsets)
        };
        Ok(SpanOut {
            qid: plan.qid.clone(),
            over: row.span_over.clone().context("span target")?,
            names: names.clone(),
            offs: plan.offsets.clone(),
            logits,
            head: row.head.clone(),
            head_reason: row.head_reason.clone(),
            label_alias: plan.label_alias.clone(),
            n_tokens: plan.target_tokens,
        })
    }
}

fn span_probabilities(span: &SpanOut, entry: &Value, calibration: &Calibration) -> Vec<f32> {
    let bounds = entry["range"]
        .as_array()
        .and_then(|bounds| Some((bounds.first()?.as_u64()?, bounds.get(1)?.as_u64()?)));
    let temperature = calibration.span_temperature(&span.head);
    span.logits
        .iter()
        .zip(&span.offs)
        .filter(|(_, (start, end))| {
            bounds.is_none_or(|(lo, hi)| *start as u64 >= lo && *end as u64 <= hi)
        })
        .map(|(logits, _)| {
            logits
                .iter()
                .map(|v| sigmoid(f64::from(*v) / temperature) as f32)
                .fold(0.0, f32::max)
        })
        .collect()
}
