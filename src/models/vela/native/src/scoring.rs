//! The scoring and decode mirror of the pinned runtime's calibration
//! path (`calibrated`, `span_threshold`, the PII length rule and sparse
//! gate, `span_decode_r4.decode_v2`, `_trim`), turning head logits into
//! predict-level answers. Parity target: the `head-input-*.json`
//! fixtures' `predict.answers`.

use serde_json::Value;

/// The package's `calibration.json` as typed lookup.
pub struct Calibration {
    pub data: Value,
}

pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|v| v / sum).collect()
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Calibration {
    pub fn load(bundle: &std::path::Path) -> anyhow::Result<Self> {
        Ok(Self {
            data: serde_json::from_slice(&std::fs::read(bundle.join("calibration.json"))?)?,
        })
    }

    fn temperature(&self, qtype: &str) -> f64 {
        self.data["temperature"]
            .get(qtype)
            .and_then(Value::as_f64)
            .unwrap_or(1.0)
    }

    pub fn span_temperature(&self, head: &str) -> f64 {
        if head == "broad" {
            self.data["broad_head"]["temperature"]
                .as_f64()
                .unwrap_or(1.0)
        } else {
            self.temperature("span")
        }
    }

    fn is_pii(&self, qid: &str, names: &[String]) -> bool {
        if qid == "pii" {
            return true;
        }
        let types: Vec<&str> = self.data["pii_types"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        !names.is_empty() && names.iter().all(|n| types.contains(&n.as_str()))
    }

    /// log-space piecewise-linear between anchors, constant outside.
    fn pii_length_threshold(&self, n_tokens: usize) -> f64 {
        let anchors = self.data["pii_length_rule"]["anchors"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut points: Vec<(f64, f64)> = anchors
            .iter()
            .map(|a| {
                (
                    a["n_tokens"].as_f64().unwrap_or(1.0).ln(),
                    a["threshold"].as_f64().unwrap_or(1.0).ln(),
                )
            })
            .collect();
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let x = (n_tokens.max(1) as f64).ln();
        let (xs, ys): (Vec<f64>, Vec<f64>) = points.into_iter().unzip();
        interp(&xs, &ys, x).exp()
    }

    /// The span threshold and its rule string, before the sparse gate.
    pub fn span_threshold(
        &self,
        qid: &str,
        names: &[String],
        n_tokens: usize,
        head: &str,
    ) -> (f64, String) {
        if head == "broad" {
            return (
                self.data["broad_head"]["threshold"].as_f64().unwrap_or(0.5),
                "broad".into(),
            );
        }
        let th = &self.data["thresholds"];
        if self.is_pii(qid, names) {
            if self
                .data
                .get("pii_length_rule")
                .is_some_and(|v| !v.is_null())
            {
                return (self.pii_length_threshold(n_tokens), "pii:length".into());
            }
            for bucket in self.data["pii_buckets"]
                .as_array()
                .cloned()
                .unwrap_or_default()
            {
                if n_tokens as f64 <= bucket["max_tokens"].as_f64().unwrap_or(f64::INFINITY) {
                    return (
                        bucket["threshold"].as_f64().unwrap_or(0.5),
                        format!("pii:{}", bucket["name"].as_str().unwrap_or("")),
                    );
                }
            }
        }
        for fam in ["halu", "toxic"] {
            if qid == fam {
                return (
                    th[&format!("span:{fam}")].as_f64().unwrap_or(0.5),
                    fam.into(),
                );
            }
        }
        let specific = format!("span:{qid}");
        if th.get(&specific).is_some() {
            (th[&specific].as_f64().unwrap_or(0.5), specific)
        } else {
            (th["span:*"].as_f64().unwrap_or(0.5), "span:*".into())
        }
    }

    pub fn set_threshold(&self, qid: &str) -> f64 {
        let th = &self.data["thresholds"];
        th.get(format!("set:{qid}"))
            .or_else(|| th.get("set:*"))
            .and_then(Value::as_f64)
            .unwrap_or(0.5)
    }

    /// The PII sparse gate (disabled when K < 0; second decode pass
    /// over the same probabilities).
    pub fn sparse_gate(
        &self,
        probs: &[Vec<f32>],
        offs: &[(usize, usize)],
        names: &[String],
        text: &str,
        thr: f64,
        rule: &str,
    ) -> (f64, String) {
        let gate = &self.data["pii_sparse_gate"];
        let k = gate["K"].as_i64().unwrap_or(-1);
        if k < 0 || thr >= gate["t_sparse"].as_f64().unwrap_or(f64::INFINITY) || offs.is_empty() {
            return (thr, rule.into());
        }
        let probe = gate["probe_threshold"].as_f64().unwrap_or(0.5);
        let decoded = decode_spans(probs, offs, names, text, probe);
        if decoded.len() as i64 <= k {
            (
                gate["t_sparse"].as_f64().unwrap_or(thr),
                format!("{rule}+sparse"),
            )
        } else {
            (thr, rule.into())
        }
    }
}

/// numpy.interp: linear between points, constant outside.
fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    for i in 0..xs.len() - 1 {
        if x >= xs[i] && x <= xs[i + 1] {
            let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
            return ys[i] + t * (ys[i + 1] - ys[i]);
        }
    }
    ys[ys.len() - 1]
}

// ------------------------------------------------------------ span decoding

const EDGE: &str = "[](){}<>「」『』【】《》〈》\"'“”‘’«»‹›";
const URL_TRAIL: &str = ".,;:!?)";

fn is_cjk(ch: char) -> bool {
    // The reference's CJK class; same ranges as the word splitter.
    super::words::is_cjk_char(ch)
}

fn trim_edges(text: &[char], s: &mut usize, e: &mut usize) {
    while *s < *e && (text[*s].is_whitespace() || EDGE.contains(text[*s])) {
        *s += 1;
    }
    while *e > *s && (text[*e - 1].is_whitespace() || EDGE.contains(text[*e - 1])) {
        *e -= 1;
    }
}

fn trim_url(text: &[char], s: &mut usize, e: &mut usize) {
    // (?:https?://|www\.) at s, case-insensitive.
    // The reference regex matches against the full text from s; the
    // lookahead is not bounded by the span end.
    let head: String = text[*s..(*s + 8).min(text.len())].iter().collect();
    let is_url = {
        let lower = head.to_lowercase();
        lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("www.")
    };
    if !is_url {
        return;
    }
    if let Some(pos) = text[*s..*e].iter().position(|&c| is_cjk(c)) {
        *e = *s + pos;
    }
    while *e > *s
        && (URL_TRAIL.contains(text[*e - 1])
            || EDGE.contains(text[*e - 1])
            || text[*e - 1].is_whitespace())
    {
        *e -= 1;
    }
}

fn trim(text: &[char], s: usize, e: usize) -> (usize, usize) {
    let (mut s, mut e) = (s, e);
    trim_edges(text, &mut s, &mut e);
    trim_url(text, &mut s, &mut e);
    (s, e)
}

/// `span_decode_r4.decode_v2` with the scorer's arguments
/// (extend=false, trim, no bridge, unit vote + fill on).
#[allow(clippy::too_many_arguments)]
pub fn decode_spans(
    probs: &[Vec<f32>],
    offs: &[(usize, usize)],
    names: &[String],
    text: &str,
    thr: f64,
) -> Vec<serde_json::Value> {
    let chars: Vec<char> = text.chars().collect();
    let t_count = offs.len();
    if t_count == 0 {
        return Vec::new();
    }
    let valid: Vec<bool> = offs.iter().map(|(a, b)| b > a).collect();
    let mut lab = vec![-1i64; t_count];
    for t in 0..t_count {
        let (k, pk) = argmax(&probs[t]);
        if valid[t] && pk as f64 > thr {
            lab[t] = k as i64;
        }
    }
    // Unit ids over the text (the reference's `_UNIT`), then per-word
    // first-nonspace lookup and alnum test.
    let units = super::words::unit_ids(&chars);
    let mut tu = vec![-1i64; t_count];
    let mut alnum = vec![false; t_count];
    for t in 0..t_count {
        if !valid[t] {
            continue;
        }
        let (a, b) = offs[t];
        let mut f = a;
        while f < b && chars[f].is_whitespace() {
            f += 1;
        }
        tu[t] = if f < b && f < chars.len() {
            units[f]
        } else {
            -1
        };
        alnum[t] = chars[a..b].iter().any(|&c| is_alnum(c));
    }
    // Unit vote + fill, per run of the same unit.
    let mut t = 0;
    while t < t_count {
        if tu[t] < 0 {
            t += 1;
            continue;
        }
        let mut t2 = t;
        while t2 + 1 < t_count && (tu[t2 + 1] == tu[t] || !valid[t2 + 1]) {
            t2 += 1;
        }
        let idx: Vec<usize> = (t..=t2).filter(|&i| valid[i]).collect();
        let labeled: Vec<usize> = idx.iter().copied().filter(|&i| lab[i] >= 0).collect();
        if !labeled.is_empty() {
            let distinct: Vec<i64> = {
                let mut d: Vec<i64> = labeled.iter().map(|&i| lab[i]).collect();
                d.sort_unstable();
                d.dedup();
                d
            };
            if distinct.len() > 1 {
                // The reference picks the first strict maximum in
                // labeled-token order; accumulate in a Vec keyed by
                // first occurrence, not a HashMap (whose iteration
                // order would make ties nondeterministic).
                let mut order: Vec<i64> = Vec::new();
                let mut scores: Vec<f64> = Vec::new();
                for &i in &labeled {
                    let label = lab[i];
                    match order.iter().position(|&l| l == label) {
                        Some(index) => scores[index] += probs[i][label as usize] as f64,
                        None => {
                            order.push(label);
                            scores.push(probs[i][label as usize] as f64);
                        }
                    }
                }
                let mut winner_index = 0usize;
                for (index, score) in scores.iter().enumerate().skip(1) {
                    if *score > scores[winner_index] {
                        winner_index = index;
                    }
                }
                let winner = order[winner_index];
                for &i in &labeled {
                    lab[i] = winner;
                }
            }
            let labels: Vec<i64> = {
                let mut d: Vec<i64> = labeled.iter().map(|&i| lab[i]).collect();
                d.sort_unstable();
                d.dedup();
                d
            };
            for c in labels {
                let pos: Vec<usize> = labeled.iter().copied().filter(|&i| lab[i] == c).collect();
                for &i in &idx {
                    if pos[0] < i && i < pos[pos.len() - 1] && lab[i] < 0 {
                        lab[i] = c;
                    }
                }
            }
        }
        t = t2 + 1;
    }
    // Merge consecutive same-label words into spans, trim, mean prob.
    let mut spans: Vec<(i64, usize, usize, Vec<usize>)> = Vec::new();
    for t in 0..t_count {
        if !valid[t] {
            continue;
        }
        let c = lab[t];
        if let Some(last) = spans.last_mut()
            && c == last.0
        {
            last.2 = offs[t].1;
            last.3.push(t);
            continue;
        }
        if c >= 0 {
            spans.push((c, offs[t].0, offs[t].1, vec![t]));
        }
    }
    let mut out = Vec::new();
    for (c, s, e, ts) in spans {
        let (s, e) = trim(&chars, s, e);
        if e <= s {
            continue;
        }
        let mean = ts.iter().map(|&t| probs[t][c as usize] as f64).sum::<f64>() / ts.len() as f64;
        out.push(serde_json::json!({
            "start": s, "end": e,
            "label": names[c as usize],
            "probability": mean,
        }));
    }
    out
}

fn argmax(row: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut value = f32::NEG_INFINITY;
    for (index, item) in row.iter().enumerate() {
        if *item > value {
            value = *item;
            best = index;
        }
    }
    (best, value)
}

fn is_alnum(c: char) -> bool {
    // The reference's `_ALNUM` = [^\W_]: word characters minus the
    // underscore — Rust's alphanumeric already excludes it.
    c.is_alphanumeric()
}

// -------------------------------------------------------- answer assembly

/// One scored question block: the reference's `raw["qs"]` entry.
pub struct QBlock {
    pub qid: String,
    pub qtype: String,
    pub names: Vec<String>,
    pub has_abs: bool,
    /// Logits over `names` (plus `[ABS]` last when `has_abs`),
    /// **after** run_batch's `set_bias` addition for set blocks.
    pub logits: Vec<f32>,
}

/// One scored span: the reference's `raw["span"]` (post aggregation).
pub struct SpanOut {
    pub qid: String,
    pub over: String,
    pub names: Vec<String>,
    pub offs: Vec<(usize, usize)>,
    pub logits: Vec<Vec<f32>>,
    pub head: String,
    pub head_reason: Option<String>,
    pub label_alias: Option<Value>,
    /// The span part's kept token count (the PII length-rule input).
    pub n_tokens: usize,
}

/// `calibrated` for one row: logits -> predict-level answers.
pub fn row_answers(
    cal: &Calibration,
    blocks: &[QBlock],
    span: Option<&SpanOut>,
    thresholds: &std::collections::HashMap<String, f64>,
    span_text: Option<&str>,
) -> Vec<Value> {
    let mut answers = Vec::new();
    for block in blocks {
        let logits: Vec<f64> = block.logits.iter().map(|v| *v as f64).collect();
        let n = block.names.len();
        let temperature = cal.temperature(&block.qtype);
        if block.qtype == "set" {
            let p: Vec<f64> = logits[..n]
                .iter()
                .map(|v| sigmoid(v / temperature))
                .collect();
            let thr = thresholds
                .get(&block.qid)
                .copied()
                .unwrap_or_else(|| cal.set_threshold(&block.qid));
            let probs: serde_json::Map<String, Value> = block
                .names
                .iter()
                .zip(&p)
                .map(|(name, value)| (name.clone(), json_f64(*value)))
                .collect();
            let selected: Vec<&String> = block
                .names
                .iter()
                .zip(&p)
                .filter(|(_, value)| **value > thr)
                .map(|(name, _)| name)
                .collect();
            answers.push(serde_json::json!({
                "id": block.qid, "type": "set",
                "answer": selected,
                "probabilities": probs,
                "threshold": thr,
            }));
            continue;
        }
        let p = softmax(
            &logits[..n]
                .iter()
                .map(|v| v / temperature)
                .collect::<Vec<_>>(),
        );
        let answer = block.names[argmax_usize(&logits[..n])].clone();
        let probs: serde_json::Map<String, Value> = block
            .names
            .iter()
            .zip(&p)
            .map(|(name, value)| (name.clone(), json_f64(*value)))
            .collect();
        let mut entry = serde_json::json!({
            "id": block.qid, "type": block.qtype,
            "answer": answer,
            "probabilities": probs,
        });
        if block.has_abs {
            let full = softmax(&logits.iter().map(|v| v / temperature).collect::<Vec<_>>());
            entry["abstain_probability"] = json_f64(full[full.len() - 1]);
        }
        if block.qtype == "score" {
            let expected: f64 = p.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
            entry["expected_index"] = json_f64(expected);
            // The default values normalize the levels to 0..1
            // (i / (n-1)); the SystemOne path never passes explicit ones.
            let n = p.len();
            let score: f64 = p
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    // The reference's default values: i/(n-1), or 0 for
                    // a single level (never NaN).
                    let value = if n > 1 {
                        i as f64 / (n - 1) as f64
                    } else {
                        0.0
                    };
                    value * v
                })
                .sum();
            entry["score"] = json_f64(score);
        }
        answers.push(entry);
    }
    if let Some(span) = span {
        let text = span_text.unwrap_or("");
        let head = span.head.as_str();
        // The per-question override wins over the calibrated rule.
        let (mut thr, mut rule) = match thresholds.get(&span.qid) {
            Some(given) => (*given, "given".to_string()),
            None => cal.span_threshold(&span.qid, &span.names, span.n_tokens, head),
        };
        let temperature = cal.span_temperature(head);
        let probs: Vec<Vec<f32>> = span
            .logits
            .iter()
            .map(|row| {
                row.iter()
                    .map(|v| sigmoid(*v as f64 / temperature) as f32)
                    .collect()
            })
            .collect();
        if rule.starts_with("pii:") {
            let (gated, gated_rule) =
                cal.sparse_gate(&probs, &span.offs, &span.names, text, thr, &rule);
            thr = gated;
            rule = gated_rule;
        }
        let mut decoded = if span.offs.is_empty() {
            Vec::new()
        } else {
            decode_spans(&probs, &span.offs, &span.names, text, thr)
        };
        let chars: Vec<char> = text.chars().collect();
        for entry in decoded.iter_mut() {
            let (s, e) = (
                entry["start"].as_u64().unwrap_or(0) as usize,
                entry["end"].as_u64().unwrap_or(0) as usize,
            );
            entry["text"] = Value::String(chars[s..e].iter().collect());
            if let Some(mapped) = span
                .label_alias
                .as_ref()
                .and_then(Value::as_object)
                .and_then(|alias| alias.get(entry["label"].as_str().unwrap_or("")))
            {
                entry["label"] = mapped.clone();
            }
        }
        let mut answer = serde_json::json!({
            "id": span.qid,
            "type": "span",
            "target_part": span.over,
            "spans": decoded,
            "threshold": thr,
            "threshold_rule": rule,
            "head": head,
        });
        if let Some(reason) = &span.head_reason {
            answer["head_reason"] = Value::String(reason.clone());
        }
        answers.push(answer);
    }
    answers
}

fn argmax_usize(x: &[f64]) -> usize {
    let mut best = 0usize;
    let mut value = f64::NEG_INFINITY;
    for (index, item) in x.iter().enumerate() {
        if *item > value {
            value = *item;
            best = index;
        }
    }
    best
}

fn json_f64(value: f64) -> Value {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// `predict_raw`'s window aggregation: sum the window rows' span logits
/// per original word (matched by char offsets), divide by coverage
/// count, uncovered words score -30.
pub fn aggregate_windows(
    names: &[String],
    original_offs: &[(usize, usize)],
    window_logits: &[Vec<Vec<f32>>],
    window_offs: &[Vec<(usize, usize)>],
) -> Vec<Vec<f32>> {
    let labels = names.len();
    // The reference accumulates in float64; the result casts to f32.
    let mut acc = vec![vec![0.0f64; labels]; original_offs.len()];
    let mut cnt = vec![0usize; original_offs.len()];
    for (logits, offs) in window_logits.iter().zip(window_offs) {
        for ((s, e), row) in offs.iter().zip(logits) {
            if let Some(u) = original_offs.iter().position(|(os, oe)| os == s && oe == e) {
                for (l, value) in row.iter().enumerate() {
                    acc[u][l] += *value as f64;
                }
                cnt[u] += 1;
            }
        }
    }
    acc.iter()
        .zip(&cnt)
        .map(|(row, c)| {
            if *c > 0 {
                row.iter().map(|v| (*v / *c as f64) as f32).collect()
            } else {
                vec![-30.0; labels]
            }
        })
        .collect()
}

// ---------------------------------------------------- SystemOne finishing

/// `system_one_response`: predict answers + the response plan -> the
/// SystemOne envelope (model, answers, usage, spans/span_heads,
/// sets, thresholds), with the span Noul view = the max word
/// probability (pre-decode, range-filtered) and field-clipped spans.
#[allow(clippy::too_many_arguments)]
pub fn system_one_response(
    model_name: &str,
    plan: &[Value],
    predict_answers: &[Value],
    span_word_probabilities: &std::collections::HashMap<String, Vec<f32>>,
    part_texts: &std::collections::HashMap<String, String>,
    input_tokens: usize,
) -> Value {
    let by_id: std::collections::HashMap<&str, &Value> = predict_answers
        .iter()
        .map(|a| (a["id"].as_str().unwrap_or(""), a))
        .collect();
    let mut answers = serde_json::Map::new();
    let mut spans = serde_json::Map::new();
    let mut span_heads = serde_json::Map::new();
    let mut sets = serde_json::Map::new();
    let mut thresholds = serde_json::Map::new();
    for entry in plan {
        let qid = entry["qid"].as_str().unwrap_or("");
        let Some(answer) = by_id.get(qid) else {
            continue;
        };
        match entry["type"].as_str().unwrap_or("") {
            "noul" => {
                answers.insert(
                    qid.to_string(),
                    serde_json::json!({"type": "noul", "noul": answer["probabilities"]["yes"]}),
                );
            }
            "choice" => {
                let names: Vec<&str> = entry["criteria"]
                    .as_object()
                    .map(|c| c.keys().map(String::as_str).collect())
                    .unwrap_or_default();
                let probabilities: serde_json::Map<String, Value> = names
                    .iter()
                    .map(|name| (name.to_string(), answer["probabilities"][*name].clone()))
                    .collect();
                let values: Vec<f64> = names
                    .iter()
                    .map(|name| answer["probabilities"][*name].as_f64().unwrap_or(0.0))
                    .collect();
                answers.insert(
                    qid.to_string(),
                    serde_json::json!({
                        "type": "choice",
                        "choice": answer["answer"],
                        "confidence": json_f64(choice_confidence(&values)),
                        "probabilities": probabilities,
                    }),
                );
            }
            "score" => {
                let levels: Vec<Value> = entry["criteria"].as_array().cloned().unwrap_or_default();
                // The predict answer keys probabilities by the level's
                // rendered name (content_text), not by index.
                let names: Vec<String> = levels
                    .iter()
                    .map(|level| match level {
                        Value::String(text) => text.clone(),
                        other => crate::state::canonical(other),
                    })
                    .collect();
                let values: Vec<f64> = names
                    .iter()
                    .map(|name| {
                        answer["probabilities"][name.as_str()]
                            .as_f64()
                            .unwrap_or(0.0)
                    })
                    .collect();
                let expected: f64 = values.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
                let legend: serde_json::Map<String, Value> = levels
                    .iter()
                    .enumerate()
                    .map(|(i, level)| (i.to_string(), level.clone()))
                    .collect();
                let probabilities: serde_json::Map<String, Value> = values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), json_f64(*v)))
                    .collect();
                answers.insert(
                    qid.to_string(),
                    serde_json::json!({
                        "type": "score",
                        "score": json_f64(expected),
                        "confidence": json_f64(score_confidence(&values)),
                        "legend": legend,
                        "probabilities": probabilities,
                    }),
                );
            }
            "set" => {
                let selected: Vec<Value> = answer["answer"].as_array().cloned().unwrap_or_default();
                sets.insert(
                    qid.to_string(),
                    serde_json::json!({
                        "selected": selected,
                        "probabilities": answer["probabilities"],
                    }),
                );
                thresholds.insert(qid.to_string(), answer["threshold"].clone());
                for (label, value) in answer["probabilities"]
                    .as_object()
                    .unwrap_or(&serde_json::Map::new())
                {
                    answers.insert(
                        format!("{qid}.{label}"),
                        serde_json::json!({"type": "noul", "noul": value}),
                    );
                }
            }
            _ => {
                // span: clip to the plan's field range and re-trim.
                let part = entry["part"].as_str().unwrap_or("user");
                let text = part_texts.get(part).cloned().unwrap_or_default();
                let chars: Vec<char> = text.chars().collect();
                let mut out = Vec::new();
                match entry["range"].as_array() {
                    Some(range) if range.len() == 2 => {
                        let (lo, hi) = (
                            range[0].as_u64().unwrap_or(0) as usize,
                            range[1].as_u64().unwrap_or(0) as usize,
                        );
                        for z in answer["spans"].as_array().unwrap_or(&vec![]) {
                            let s = z["start"].as_u64().unwrap_or(0) as usize;
                            let e = z["end"].as_u64().unwrap_or(0) as usize;
                            let (s, e) = trim(&chars, s.max(lo), e.min(hi));
                            if e > s {
                                out.push(serde_json::json!({
                                    "label": z["label"],
                                    "start": s - lo,
                                    "end": e - lo,
                                    "text": chars[s..e].iter().collect::<String>(),
                                    "probability": z["probability"],
                                }));
                            }
                        }
                    }
                    _ => {
                        out = answer["spans"].as_array().cloned().unwrap_or_default();
                    }
                }
                spans.insert(qid.to_string(), Value::Array(out));
                span_heads.insert(qid.to_string(), answer["head"].clone());
                thresholds.insert(qid.to_string(), answer["threshold"].clone());
                let max = span_word_probabilities
                    .get(qid)
                    .map(|probs| probs.iter().cloned().fold(0.0f32, f32::max))
                    .unwrap_or(0.0);
                answers.insert(
                    qid.to_string(),
                    serde_json::json!({"type": "noul", "noul": json_f64(max as f64)}),
                );
            }
        }
    }
    let mut response = serde_json::Map::new();
    response.insert("model".into(), Value::String(model_name.to_string()));
    response.insert("answers".into(), Value::Object(answers));
    response.insert(
        "usage".into(),
        serde_json::json!({"input_tokens": input_tokens, "output_tokens": 0}),
    );
    if !spans.is_empty() {
        response.insert("spans".into(), Value::Object(spans));
        response.insert("span_heads".into(), Value::Object(span_heads));
    }
    if !sets.is_empty() {
        response.insert("sets".into(), Value::Object(sets));
    }
    if !thresholds.is_empty() {
        response.insert("thresholds".into(), Value::Object(thresholds));
    }
    Value::Object(response)
}

/// The top-two probability margin (the reference's `_choice_confidence`).
pub fn choice_confidence(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    if sorted.len() == 1 {
        return 1.0;
    }
    (sorted[0] - sorted[1]).clamp(0.0, 1.0)
}

/// One minus the normalized level variance (`_score_confidence`).
pub fn score_confidence(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 1 {
        return 1.0;
    }
    let mean: f64 = values.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
    let var: f64 = values
        .iter()
        .enumerate()
        .map(|(i, v)| v * (i as f64 - mean).powi(2))
        .sum();
    (1.0 - var / ((n * n - 1) as f64 / 12.0)).clamp(0.0, 1.0)
}
