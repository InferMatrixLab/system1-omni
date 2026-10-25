//! Prompt rendering, mirroring the pinned runtime's `_prep` + `_render`
//! (the DEV2.0 decoder layout): the parts pass, per-question blocks
//! with option end and query positions, the span block in layout a/c
//! with word positions, the truncation budget, and the long-target
//! windows. Output structs serialize exactly like the frozen
//! `native-plan-*` fixtures.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use std::collections::BTreeMap;
use tokenizers::Tokenizer;

use crate::contract::RequestError;
use crate::state::canonical;

pub const SUFFIX_CHOICE_SCORE: &str =
    "\n\nSelect the single option best supported by the context and instructions.\nDecision:";
pub const SUFFIX_SET: &str =
    "\n\nSelect every option supported by the context and instructions.\nDecision:";
pub const SUFFIX_SPAN: &str =
    "\n\nMark every span of the target text that carries one of the labels.\nSpans:";
pub const ABS_KEY: &str = "[ABS]";
pub const ABS_DESC: &str = "None of the shown options applies.";
pub const PARTS_HEAD: &str = "Context:\n";
pub const SEG_CLOSE: &str = "\n</segment>\n";
pub const TEXT_HEAD: &str = "\n\nText:\n";
pub const HYBRID_RMAX: usize = 2048;
pub const WIN: usize = 1800;
pub const STRIDE: usize = 1536;
pub const RMAX: usize = 2048;

pub const ROUTER_SPAN_IDS: [&str; 3] = ["pii", "halu", "toxic"];
pub const HALU_LABEL: &str = "unsupported";
pub const HALU_ALIASES: [&str; 9] = [
    "unsupported",
    "unsupportedclaim",
    "hallucinated",
    "hallucination",
    "hallucinatedspan",
    "notsupported",
    "unsupportedspan",
    "fabricated",
    "fabrication",
];
pub const PII_TYPES: [&str; 17] = [
    "AGE",
    "CREDIT_CARD",
    "DATE_TIME",
    "DOMAIN_NAME",
    "EMAIL_ADDRESS",
    "GPE",
    "IBAN_CODE",
    "IP_ADDRESS",
    "NRP",
    "ORGANIZATION",
    "PERSON",
    "PHONE_NUMBER",
    "STREET_ADDRESS",
    "TITLE",
    "US_DRIVER_LICENSE",
    "US_SSN",
    "ZIP_CODE",
];

fn seg_open(role: &str) -> String {
    format!("<segment role=\"{role}\">\n")
}

fn option_block(key: &str, desc: &str, tag: &str) -> String {
    format!(
        "\n<{tag}>\n{}\n</{tag}>",
        canonical(&serde_json::json!({"key": key, "description": desc}))
    )
}

fn q_head(kind: &str, over: &[String], text: &str, span: bool) -> String {
    let target = over.join(", ");
    format!(
        "\n\nTask type: {kind}\nTarget: {target}\nQuestion:\n{text}\n{}",
        if span { "Labels:" } else { "Options:" }
    )
}

// ------------------------------------------------------------------ tokenizer

/// The pinned tokenizer, with the reference's memoized plain-`ids` view
/// and char-offset encodings (the tokenizers crate reports byte
/// offsets; the reference works in Unicode code points).
/// Ids plus char offsets, the encoder cache entry.
type Encoded = (Vec<u32>, Vec<(usize, usize)>);

pub struct Encoder {
    tokenizer: Tokenizer,
    cache: RefCell<HashMap<String, Encoded>>,
}

impl Encoder {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(anyhow::Error::msg)?;
        Ok(Self {
            tokenizer,
            cache: RefCell::new(HashMap::new()),
        })
    }

    fn encode(&self, text: &str) -> (Vec<u32>, Vec<(usize, usize)>) {
        if let Some(hit) = self.cache.borrow().get(text) {
            return hit.clone();
        }
        let encoding = match self.tokenizer.encode(text, false) {
            Ok(encoding) => encoding,
            Err(error) => {
                // The pinned tokenizer should never fail on valid UTF-8
                // input; surface the error instead of panicking so a
                // wrapping worker can map it.
                unreachable!("the pinned tokenizer failed on valid input: {error}")
            }
        };
        let byte_offsets = encoding.get_offsets();
        // Byte → code-point offsets.
        let char_starts: Vec<usize> = text
            .char_indices()
            .map(|(byte, _)| byte)
            .chain(std::iter::once(text.len()))
            .collect();
        let to_char = |byte: usize| char_starts.partition_point(|&start| start < byte);
        let offsets: Vec<(usize, usize)> = byte_offsets
            .iter()
            .map(|(start, end)| (to_char(*start), to_char(*end)))
            .collect();
        let entry = (encoding.get_ids().to_vec(), offsets);
        self.cache
            .borrow_mut()
            .insert(text.to_owned(), entry.clone());
        entry
    }

    /// Memoized token ids of a whole string (the reference's `_ids`).
    pub fn ids(&self, text: &str) -> Vec<u32> {
        self.encode(text).0
    }

    /// Ids and char offsets (the reference's `_ids_offs`).
    pub fn ids_offsets(&self, text: &str) -> (Vec<u32>, Vec<(usize, usize)>) {
        self.encode(text)
    }
}

// ------------------------------------------------------------- span dispatch

/// `[\s_\-]+` removed, lower-cased — the reference's `_norm_label`.
pub fn norm_label(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if !ch.is_whitespace() && ch != '_' && ch != '-' {
            out.extend(ch.to_lowercase());
        }
    }
    out
}

/// The user's label when the span question has exactly one label that
/// is a hallucination alias, else none.
pub fn halu_alias(names: &[String]) -> Option<String> {
    if names.len() == 1 && HALU_ALIASES.contains(&norm_label(&names[0]).as_str()) {
        Some(names[0].clone())
    } else {
        None
    }
}

pub struct SpanDispatch {
    pub head: &'static str,
    pub reason: &'static str,
}

/// The reference's six-rule dispatch, in order: `"head"` override,
/// trained ids, an exact router label set, a non-empty subset of the 17
/// PII types, one hallucination alias, else broad (router when the
/// broad head is not loaded).
pub fn span_head_for(
    id: &str,
    labels: &HashSet<String>,
    head_override: Option<&str>,
    router_label_sets: &[HashSet<String>],
    broad_available: bool,
) -> Result<SpanDispatch, String> {
    if let Some(head) = head_override {
        if !matches!(head, "router" | "broad") {
            return Err(format!(
                "head must be one of ['router', 'broad'], got '{head:?}'"
            ));
        }
        if head == "broad" && !broad_available {
            return Err("the broad span head is not loaded (needs broad_head.safetensors)".into());
        }
        return Ok(SpanDispatch {
            head: if head == "broad" { "broad" } else { "router" },
            reason: "override",
        });
    }
    if ROUTER_SPAN_IDS.contains(&id) {
        return Ok(SpanDispatch {
            head: "router",
            reason: "id",
        });
    }
    if router_label_sets.iter().any(|set| set == labels) {
        return Ok(SpanDispatch {
            head: "router",
            reason: "label_set",
        });
    }
    let pii: HashSet<&str> = PII_TYPES.into_iter().collect();
    if !labels.is_empty() && labels.iter().all(|label| pii.contains(label.as_str())) {
        return Ok(SpanDispatch {
            head: "router",
            reason: "pii_subset",
        });
    }
    if halu_alias(&labels.iter().cloned().collect::<Vec<_>>()).is_some() {
        return Ok(SpanDispatch {
            head: "router",
            reason: "halu_alias",
        });
    }
    if !broad_available {
        return Ok(SpanDispatch {
            head: "router",
            reason: "broad_unavailable",
        });
    }
    Ok(SpanDispatch {
        head: "broad",
        reason: "default",
    })
}

// ------------------------------------------------------------------ rendering

/// One generic-contract question at render time (Noul is already folded
/// into a two-option choice; choice always carries `[ABS]` on the
/// SystemOne path — the reference's `_to_generic` defaults abstain to
/// true and the mapper never sets it).
#[derive(Clone)]
pub struct GenericQuestion {
    pub id: String,
    pub kind: String,
    pub over: Vec<String>,
    pub text: String,
    pub names: Vec<String>,
    pub descriptions: Vec<String>,
    pub threshold: Option<f64>,
    pub head_override: Option<String>,
}

/// One prepared part: tokens, and for span-target parts also char
/// offsets and word units.
/// Word char offsets and each word's first token index.
pub type WordUnits = (Vec<(usize, usize)>, Vec<usize>);

#[derive(Clone)]
pub struct PreparedPart {
    pub role: String,
    pub ids: Vec<u32>,
    pub offsets: Option<Vec<(usize, usize)>>,
    pub words: Option<WordUnits>,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct TruncRecord {
    pub shrink_cut: usize,
    pub protected_cut: usize,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
#[serde(tag = "kind")]
pub enum Block {
    #[serde(rename = "q")]
    Question {
        ids: Vec<u32>,
        ends: Vec<usize>,
        query: usize,
        qid: String,
        qtype: String,
        names: Vec<String>,
        has_abs: bool,
    },
    #[serde(rename = "span")]
    Span {
        ends: Vec<usize>,
        label_ranges: Vec<(usize, usize)>,
        words_in_block: bool,
        ids: Vec<u32>,
        query: usize,
        w_pos: Vec<usize>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderedRow {
    pub p_ids: Vec<u32>,
    pub blocks: Vec<Block>,
    pub trunc: TruncRecord,
    pub seg_lens: BTreeMap<String, usize>,
    pub layout: Option<String>,
    pub names: Option<Vec<String>>,
    pub w_offs: Option<Vec<(usize, usize)>>,
    pub w_pos: Option<Vec<usize>>,
    pub span_over: Option<String>,
    pub n_tokens: usize,
    pub head: String,
    pub head_reason: Option<String>,
    /// The runtime's window rows carry the head only — no head_reason
    /// key at all (predict_raw assigns head alone).
    pub window: bool,
}

/// Serialize exactly like the frozen fixtures: main rows always carry
/// `head_reason` (null when the row has no span); window rows omit it.
impl serde::Serialize for RenderedRow {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serialize_row(self, serializer)
    }
}

fn serialize_row<S>(row: &RenderedRow, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(None)?;
    macro_rules! put {
        ($key:literal, $value:expr) => {
            map.serialize_entry($key, &$value)?
        };
    }
    put!("p_ids", row.p_ids);
    put!("blocks", row.blocks);
    put!("trunc", row.trunc);
    put!("seg_lens", row.seg_lens);
    put!("layout", row.layout);
    if let Some(names) = &row.names {
        put!("names", names);
    }
    if let Some(w_offs) = &row.w_offs {
        put!("w_offs", w_offs);
    }
    if let Some(w_pos) = &row.w_pos {
        put!("w_pos", w_pos);
    }
    if let Some(span_over) = &row.span_over {
        put!("span_over", span_over);
    }
    put!("n_tokens", row.n_tokens);
    put!("head", row.head);
    if !row.window {
        put!("head_reason", row.head_reason);
    }
    map.end()
}

/// Everything the renderer needs beyond tokens.
pub struct RenderContext<'a> {
    pub encoder: &'a Encoder,
    pub max_length: usize,
    pub router_label_sets: &'a [HashSet<String>],
    pub has_broad: bool,
    pub halu_desc: String,
}

/// The whole-request 413: questions plus schema alone do not fit.
pub fn too_long(max_length: usize) -> RequestError {
    RequestError {
        status: 413,
        body: serde_json::json!({"detail": [{"loc": ["body", "questions"],
            "msg": format!("the questions alone do not fit the {max_length}-token window"),
            "type": "too_long"}]}),
    }
}

/// `schema_lib_r4.assemble`, eval path: the truncation order on the
/// decoder's token costs. Returns kept segment lengths and the record.
fn assemble_budget(
    q_lens: &[usize],
    el_lens: &[usize],
    segs: &mut [(String, usize)],
    max_len: usize,
    protected: &HashSet<String>,
    shrink: Option<&str>,
) -> Result<(BTreeMap<String, usize>, TruncRecord), ()> {
    let fixed = 2 + 1 + segs.len();
    fn over(
        q_lens: &[usize],
        el_lens: &[usize],
        segs: &[(String, usize)],
        fixed: usize,
        max_len: usize,
    ) -> i64 {
        fixed as i64
            + q_lens.iter().sum::<usize>() as i64
            + el_lens.iter().sum::<usize>() as i64
            + segs.iter().map(|(_, n)| *n).sum::<usize>() as i64
            - max_len as i64
    }
    let mut trunc = TruncRecord {
        shrink_cut: 0,
        protected_cut: 0,
    };
    let shrink_index = shrink.and_then(|role| segs.iter().position(|(r, _)| r == role));
    for floor in [64, 1] {
        if over(q_lens, el_lens, segs, fixed, max_len) > 0
            && let Some(index) = shrink_index
        {
            let full = segs[index].1;
            let excess = over(q_lens, el_lens, segs, fixed, max_len).max(0) as usize;
            let keep = full.min(floor.max(full.saturating_sub(excess)));
            trunc.shrink_cut += full - keep;
            segs[index].1 = keep;
        }
    }
    if over(q_lens, el_lens, segs, fixed, max_len) > 0 {
        for index in 0..segs.len() {
            if over(q_lens, el_lens, segs, fixed, max_len) > 0 && protected.contains(&segs[index].0)
            {
                let full = segs[index].1;
                let excess = over(q_lens, el_lens, segs, fixed, max_len).max(0) as usize;
                let keep = 1.max(full.saturating_sub(excess));
                trunc.protected_cut += full - keep;
                segs[index].1 = keep;
            }
        }
    }
    if over(q_lens, el_lens, segs, fixed, max_len) > 0 {
        return Err(());
    }
    let seg_lens: BTreeMap<String, usize> =
        segs.iter().map(|(role, n)| (role.clone(), *n)).collect();
    Ok((seg_lens, trunc))
}

/// Token windows [a, b) over a part of n tokens, starting and cut at
/// word starts (`schema_lib_d3_20261001.windows`).
pub fn span_windows(first_tokens: &[usize], n: usize) -> Vec<(usize, usize)> {
    let mut starts: Vec<usize> = first_tokens.to_vec();
    starts.sort_unstable();
    starts.dedup();
    if starts.is_empty() {
        return vec![(0, n)];
    }
    let mut out = Vec::new();
    let mut a = starts[0];
    loop {
        let limit = a + WIN;
        let next: Vec<usize> = starts
            .iter()
            .copied()
            .filter(|&s| s > a && s < limit)
            .collect();
        let b = if limit >= n {
            n
        } else if !next.is_empty() {
            *starts
                .iter()
                .filter(|&&s| s < limit && s > a)
                .max()
                .unwrap()
        } else {
            n.min(limit)
        };
        out.push((a, b));
        if b >= n {
            break;
        }
        let candidate: Vec<usize> = starts
            .iter()
            .copied()
            .filter(|&s| s >= a + STRIDE && s < b)
            .collect();
        a = candidate.first().copied().unwrap_or(b);
    }
    out
}

/// A windowed copy of a prepared part: tokens [a, b) and the words that
/// start inside them.
pub fn window_part(part: &PreparedPart, a: usize, b: usize) -> PreparedPart {
    let (word_offsets, first_tokens) = part.words.as_ref().expect("windowed parts carry words");
    let selected: Vec<usize> = first_tokens
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t >= a && t < b)
        .map(|(index, _)| index)
        .collect();
    let kept_offsets: Vec<(usize, usize)> = selected.iter().map(|&i| word_offsets[i]).collect();
    PreparedPart {
        role: part.role.clone(),
        ids: part.ids[a..b].to_vec(),
        offsets: part.offsets.as_ref().map(|offs| offs[a..b].to_vec()),
        words: Some((
            kept_offsets,
            selected.iter().map(|&i| first_tokens[i] - a).collect(),
        )),
    }
}

impl RenderContext<'_> {
    /// `_prep`'s question side: the span-head dispatch and the
    /// hallucination-alias label replacement (parts arrive prepared —
    /// word units need the original text and are the caller's job).
    pub fn prepare_questions(
        &self,
        questions: &[GenericQuestion],
    ) -> Result<(Vec<GenericQuestion>, SpanDispatch), RequestError> {
        let span = questions.iter().find(|q| q.kind == "span");
        let dispatch = match span {
            Some(question) => {
                let labels: HashSet<String> = question.names.iter().cloned().collect();
                span_head_for(
                    &question.id,
                    &labels,
                    question.head_override.as_deref(),
                    self.router_label_sets,
                    self.has_broad,
                )
                .map_err(|message| crate::contract::runtime_rejection(&message))?
            }
            None => SpanDispatch {
                head: "router",
                reason: "",
            },
        };
        // The trained `unsupported` label renders in place of a single
        // hallucination-alias label (the answer gets the user's name).
        let mut adjusted = questions.to_vec();
        if let Some(question) = adjusted.iter_mut().find(|q| q.kind == "span")
            && dispatch.head == "router"
            && let Some(alias) = halu_alias(&question.names)
            && alias != HALU_LABEL
        {
            question.names = vec![HALU_LABEL.to_string()];
            question.descriptions = vec![self.halu_desc.clone()];
        }
        Ok((adjusted, dispatch))
    }

    /// `_render` on one prepared row: the parts pass and every block,
    /// with the budget applied first.
    pub fn render(
        &self,
        parts: &[PreparedPart],
        questions: &[GenericQuestion],
        dispatch: &SpanDispatch,
    ) -> Result<RenderedRow, RequestError> {
        let ids_of = |text: &str| self.encoder.ids(text);
        let roles: Vec<String> = parts.iter().map(|p| p.role.clone()).collect();
        // Question blocks and their schema costs.
        struct RenderedQ {
            qid: String,
            qtype: String,
            names: Vec<String>,
            otoks: Vec<Vec<u32>>,
            has_abs: bool,
            head: Vec<u32>,
            suffix: Vec<u32>,
            abstain_tokens: Vec<u32>,
            len: usize,
        }
        let mut rendered_qs: Vec<RenderedQ> = Vec::new();
        let mut el: Vec<(String, Vec<u32>)> = Vec::new(); // span labels
        let mut span_question: Option<&GenericQuestion> = None;
        for question in questions {
            if question.kind == "span" {
                span_question = Some(question);
                for (name, desc) in question.names.iter().zip(&question.descriptions) {
                    let tokens = ids_of(&option_block(name, desc, "label"));
                    el.push((name.clone(), tokens.clone()));
                }
                continue;
            }
            let suffix = match question.kind.as_str() {
                "set" => SUFFIX_SET,
                _ => SUFFIX_CHOICE_SCORE, // choice and score share the wording
            };
            let has_abs = question.kind == "choice"; // SystemOne: abstain defaults true
            let head = ids_of(&q_head(
                &question.kind,
                &question.over,
                &question.text,
                false,
            ));
            let suffix = ids_of(suffix);
            let abstain_tokens = if has_abs {
                ids_of(&option_block(ABS_KEY, ABS_DESC, "option"))
            } else {
                Vec::new()
            };
            let mut otoks = Vec::with_capacity(question.names.len());
            let mut len =
                1 + head.len() + suffix.len() + abstain_tokens.len() + usize::from(has_abs);
            for (name, desc) in question.names.iter().zip(&question.descriptions) {
                let tokens = ids_of(&option_block(name, desc, "option"));
                len += 1 + tokens.len();
                otoks.push(tokens);
            }
            rendered_qs.push(RenderedQ {
                qid: question.id.clone(),
                qtype: question.kind.clone(),
                names: question.names.clone(),
                otoks,
                has_abs,
                head,
                suffix,
                abstain_tokens,
                len,
            });
        }
        let span_over: Option<String> = span_question.map(|question| question.over[0].clone());
        let mut overs: HashSet<String> = HashSet::new();
        for question in questions {
            overs.extend(question.over.iter().cloned());
        }
        if let Some(over) = &span_over {
            overs.insert(over.clone());
        }
        let part_of = |role: &str| parts.iter().find(|p| p.role == role);
        let shrink = roles.iter().find(|role| !overs.contains(*role)).cloned();
        let layout = match &span_over {
            Some(over) if part_of(over).is_some_and(|p| p.ids.len() <= HYBRID_RMAX) => "c",
            Some(_) => "a",
            None => "",
        };
        // The schema overhead the budget does not see.
        let mut budget = ids_of(PARTS_HEAD).len();
        for role in &roles {
            budget += ids_of(&seg_open(role)).len() + ids_of(SEG_CLOSE).len();
        }
        if let Some(over) = &span_over {
            let question = span_question.expect("span question");
            let head = ids_of(&q_head("span", &question.over, &question.text, true));
            let suffix = ids_of(SUFFIX_SPAN);
            budget += head.len() + suffix.len();
            if layout == "c" {
                let span_len = part_of(over).expect("span part").ids.len();
                budget += ids_of(TEXT_HEAD).len()
                    + ids_of(&seg_open(over)).len()
                    + ids_of(SEG_CLOSE).len();
                let headroom = self.max_length.saturating_sub(budget);
                budget += span_len.min(headroom / 2);
            }
        }
        let q_lens: Vec<usize> = rendered_qs.iter().map(|q| q.len).collect();
        let el_lens: Vec<usize> = el.iter().map(|(_, tokens)| 1 + tokens.len()).collect();
        let mut segs: Vec<(String, usize)> = parts
            .iter()
            .map(|p| (p.role.clone(), p.ids.len()))
            .collect();
        let (seg_lens, trunc) = assemble_budget(
            &q_lens,
            &el_lens,
            &mut segs,
            64.max(self.max_length - budget),
            &overs,
            shrink.as_deref(),
        )
        .map_err(|()| too_long(self.max_length))?;
        // The parts pass, in part order (not the sorted map).
        let mut p_ids = ids_of(PARTS_HEAD);
        let mut start: HashMap<String, usize> = HashMap::new();
        for part in parts {
            let keep = seg_lens[&part.role];
            let open = ids_of(&seg_open(&part.role));
            start.insert(part.role.clone(), p_ids.len() + open.len());
            p_ids.extend(open);
            p_ids.extend(part.ids.iter().take(keep));
            p_ids.extend(ids_of(SEG_CLOSE));
        }
        // Question blocks.
        let mut blocks: Vec<Block> = Vec::new();
        for q in &rendered_qs {
            let mut ids = q.head.clone();
            let mut ends = Vec::with_capacity(q.names.len() + usize::from(q.has_abs));
            for tokens in &q.otoks {
                ids.extend(tokens);
                ends.push(ids.len() - 1);
            }
            if q.has_abs {
                ids.extend(q.abstain_tokens.clone());
                ends.push(ids.len() - 1);
            }
            ids.extend(q.suffix.clone());
            let query = ids.len() - 1;
            blocks.push(Block::Question {
                ids,
                ends,
                query,
                qid: q.qid.clone(),
                qtype: q.qtype.clone(),
                names: q.names.clone(),
                has_abs: q.has_abs,
            });
        }
        // The span block.
        let mut row_names = None;
        let mut row_w_offs = None;
        let mut row_w_pos = None;
        if let Some(over) = &span_over
            && !el.is_empty()
        {
            let question = span_question.expect("span question");
            let mut ids = ids_of(&q_head("span", &question.over, &question.text, true));
            let mut ends = Vec::new();
            let mut starts = Vec::new();
            for (_, tokens) in &el {
                starts.push(ids.len());
                ids.extend(tokens);
                ends.push(ids.len() - 1);
            }
            let part = part_of(over).expect("span part");
            let n = seg_lens[over];
            let (word_offsets, first_tokens) = part.words.as_ref().expect("span words");
            // Word indices whose first token survives the kept prefix;
            // indices, not token values — two words can share one.
            let keep: Vec<usize> = first_tokens
                .iter()
                .enumerate()
                .filter(|&(_, &t)| t < n)
                .map(|(index, _)| index)
                .collect();
            let words_in_block = layout == "c";
            if words_in_block {
                ids.extend(ids_of(TEXT_HEAD));
                ids.extend(ids_of(&seg_open(over)));
                let c0 = ids.len();
                ids.extend(part.ids.iter().take(n));
                ids.extend(ids_of(SEG_CLOSE));
                row_w_pos = Some(keep.iter().map(|&i| c0 + first_tokens[i]).collect());
            } else {
                let base = start[over];
                row_w_pos = Some(keep.iter().map(|&i| base + first_tokens[i]).collect());
            }
            ids.extend(ids_of(SUFFIX_SPAN));
            let query = ids.len() - 1;
            row_names = Some(el.iter().map(|(name, _)| name.clone()).collect());
            row_w_offs = Some(keep.iter().map(|&i| word_offsets[i]).collect());
            let label_ranges: Vec<(usize, usize)> =
                starts.into_iter().zip(ends.iter().copied()).collect();
            blocks.push(Block::Span {
                ends,
                label_ranges,
                words_in_block,
                ids,
                query,
                w_pos: row_w_pos.clone().expect("w_pos"),
            });
        }
        let n_tokens = p_ids.len() + blocks.iter().map(block_len).sum::<usize>();
        Ok(RenderedRow {
            p_ids,
            blocks,
            trunc,
            seg_lens,
            layout: if span_over.is_some() {
                Some(layout.to_string())
            } else {
                None
            },
            names: row_names,
            w_offs: row_w_offs,
            w_pos: row_w_pos,
            span_over,
            n_tokens,
            head: dispatch.head.to_string(),
            head_reason: if dispatch.reason.is_empty() {
                None
            } else {
                Some(dispatch.reason.to_string())
            },
            window: false,
        })
    }
}

fn block_len(block: &Block) -> usize {
    match block {
        Block::Question { ids, .. } => ids.len(),
        Block::Span { ids, .. } => ids.len(),
    }
}
