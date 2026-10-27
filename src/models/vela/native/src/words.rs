//! Word units and first-token mapping, mirroring the pinned runtime's
//! `schema_lib_r4` `_WORD` splitter (GLiNER2 whitespace splitter + CJK
//! one unit per character + the r4 URL/email/handle rule) and
//! `word_first_tokens`. Offsets are Unicode code points everywhere, as
//! on the reference.

use onig::Regex;

/// The CJK ranges of the reference `_WORD` pattern, verbatim.
const CJK: &str = "⺀-⿿　-〿぀-ヿ㄀-ㇿ㐀-䶿一-鿿ꀀ-꓏가-힯豈-﫿︰-﹏＀-￯";
/// Trailing punctuation the URL rule strips.
const TRAIL: &str = ".,;:!?)\\]}>\"'\u{201d}\u{2019}\u{bb}";

fn word_pattern() -> Regex {
    let pattern = concat!(
        "(?:https?://|www\\.)[^\\s{cjk}]+?(?=[{trail}]*(?![^\\s{cjk}]))",
        "|[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}",
        "|@[a-z0-9_]+",
        "|[{cjk}]",
        "|[^\\W{cjk}]+(?:[-_][^\\W{cjk}]+)*",
        "|\\S",
    );
    let pattern = pattern.replace("{cjk}", CJK).replace("{trail}", TRAIL);
    Regex::with_options(
        &pattern,
        onig::RegexOptions::REGEX_OPTION_IGNORECASE,
        onig::Syntax::ruby(),
    )
    .expect("the reference word pattern must compile")
}

/// Start-of-char byte offsets plus the end sentinel; index it with
/// `partition_point(|&start| start < byte)` to map a byte offset to its
/// code-point offset.
fn char_starts(text: &str) -> Vec<usize> {
    text.char_indices()
        .map(|(byte, _)| byte)
        .chain(std::iter::once(text.len()))
        .collect()
}

/// The (start, end) word ranges of `text`, in code points.
pub fn split_words(text: &str) -> Vec<(usize, usize)> {
    let regex = word_pattern();
    // Byte offsets → code-point offsets.
    let char_of_byte = char_starts(text);
    let to_char = |byte: usize| -> usize { char_of_byte.partition_point(|&start| start < byte) };
    regex
        .find_iter(text)
        .map(|(start, end)| (to_char(start), to_char(end)))
        .collect()
}

/// (word char offsets (W, 2), each word's first overlapping sub-word
/// token index), mirroring `word_first_tokens`: the first token whose
/// cumulative end passes the word start, skipping empty tokens.
pub fn word_first_tokens(
    text: &str,
    offsets: &[(usize, usize)],
) -> (Vec<(usize, usize)>, Vec<usize>) {
    let words = split_words(text);
    if words.is_empty() || offsets.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let valid: Vec<bool> = offsets.iter().map(|(s, e)| e > s).collect();
    let mut cummax_end: Vec<i64> = Vec::with_capacity(offsets.len());
    let mut running: i64 = -1;
    for (index, (_start, end)) in offsets.iter().enumerate() {
        if valid[index] {
            running = running.max(*end as i64);
        }
        cummax_end.push(running);
    }
    let mut word_offsets = Vec::new();
    let mut first_tokens = Vec::new();
    for (a, b) in words {
        let mut t = cummax_end.partition_point(|&end| end <= a as i64);
        while t < offsets.len() && !valid[t] {
            t += 1;
        }
        if t < offsets.len() && offsets[t].0 < b {
            word_offsets.push((a, b));
            first_tokens.push(t);
        }
    }
    (word_offsets, first_tokens)
}

/// The reference's CJK class as a range test (the same ranges the word
/// splitter carries).
pub fn is_cjk_char(ch: char) -> bool {
    // The reference class has one contiguous run from U+8C48 to
    // U+FAFF and ends the Yi block at U+A4CF (not U+A48F).
    matches!(ch as u32,
        0x2E80..=0x2FFF | 0x3000..=0x303F | 0x3040..=0x30FF | 0x3100..=0x31FF
        | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xA000..=0xA4CF | 0xAC00..=0xD7AF
        | 0x8C48..=0xFAFF | 0xFE30..=0xFE4F | 0xFF00..=0xFFEF)
}

/// The reference's `_UNIT` split (decode-time units): URL bodies, runs
/// of word characters (excluding CJK, `_` excluded by `[^\W_]`) plus
/// `@._-+`, or any non-space char. Returns each char's unit id, -1
/// outside units — the decode's `u` array.
pub fn unit_ids(chars: &[char]) -> Vec<i64> {
    let mut ids = vec![-1i64; chars.len()];
    let text: String = chars.iter().collect();
    let pattern = concat!(
        "(?:https?://|www\\.)[^\\s\\]\\)\\}>「」『』\"'“”{cjk}]+?(?=[.,;:!?]*(?![^\\s\\]\\)\\}>「」『』\"'“”{cjk}]))",
        "|(?:(?![{cjk}])[^\\W_]|[@._\\-+])+",
        "|\\S",
    )
    .replace("{cjk}", CJK);
    let regex = Regex::with_options(
        &pattern,
        onig::RegexOptions::REGEX_OPTION_IGNORECASE,
        onig::Syntax::ruby(),
    )
    .expect("the reference unit pattern must compile");
    // find_iter yields byte offsets; ids indexes characters.
    let char_of_byte = char_starts(&text);
    let to_char = |byte: usize| -> usize { char_of_byte.partition_point(|&start| start < byte) };
    for (unit, (start, end)) in regex.find_iter(&text).enumerate() {
        let (start, end) = (to_char(start), to_char(end));
        for id in ids.iter_mut().take(end).skip(start) {
            *id = unit as i64;
        }
    }
    ids
}
