//! Native request processing for Valen-Preview-0923: the wire contract and the
//! pinned compiler's branch construction, equivalent to the Python reference
//! worker's `protocol` / `preprocess` layers and the pinned
//! `valen.data.compiler.Compiler` (revision 750bfcf).
//!
//! Layering mirrors the reference worker and leaves seams for the executor
//! (ticket N4):
//!
//! - [`wire`]: the JSON body decode with the reference's exact rejection
//!   bodies, plus the small Python-behaviour helpers (repr, trim).
//! - [`protocol`]: `parse_body` / `parse_request` parity — the whole 4xx
//!   surface of `src/models/valen/protocol.py`.
//! - [`preprocess`]: the reference worker's materialization semantics — the
//!   question records (`question_record`), the chat-template-equivalent state
//!   rendering, and the image grid (`Qwen2VLImageProcessor` smart resize).
//! - [`compile`]: branch suffix construction, candidate-end and decision-end
//!   position capture, `max_length` enforcement and token accounting; the
//!   token source is an [`compile::Encoder`] so the layout logic is testable
//!   CPU-only, with the real `tokenizers` backend behind [`Processor`].
//!
//! Execution (forward passes, the decision head) and the HTTP worker live in
//! [`engine`]/[`text`]/[`head`]/[`server`]; [`finish`] serves all three wire
//! question types' finishing variants — `choice`, `noul` (P(true) + the full
//! pair) and `score` (cross-branch softmax → expected level + legend +
//! concentration confidence). The unmerged text-path LoRA application
//! (spec 03, N2) is [`text`]: export-policy validation over the five-file
//! export plus the shared `#143` machinery.

pub mod compile;
pub mod engine;
pub mod finish;
pub mod head;
pub mod preprocess;
pub mod protocol;
pub mod server;
pub mod text;
pub mod wire;

pub use compile::{Branch, Compiled, CompiledQuestion, Processor};
pub use protocol::{Kind, Request, State};
