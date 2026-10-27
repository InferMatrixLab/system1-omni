//! Rust HTTP server for Clef and Clef-flash.
//!
//! `omni-clef` admits one complete request at a time through
//! [`omni_runtime::SerialScheduler`] and forwards the raw body to a single
//! Python executor process. That process runs the pinned joint-schema
//! checkpoint on its main thread, which is the thread that initialized
//! CUDA. The joint head is still the checkpoint module. It is not the
//! shared Qwen CUDA executor.
//!
//! The in-process frontend engine speaks a different error envelope, so
//! this server forwards the executor's status and JSON bytes itself.

mod cli;
mod executor;
mod frame;
mod server;

pub use cli::{
    Cli, ServeArgs, executor_python_from, locate_pythonpath_from, parse_cli, python_worker_command,
    serve_from_env,
};
pub use server::{Engine, MAX_BODY, router, serve_on};
