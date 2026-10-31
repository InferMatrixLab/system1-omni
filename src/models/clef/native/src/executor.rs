//! One executor process for the life of the server.
//!
//! PyTorch CUDA state is thread-local. `ThreadingHTTPServer` created a new
//! thread per request, so the first CUDA call on that thread paid the
//! context-init cost. This process is spawned once. Its main thread loads
//! the checkpoint, warms up, and scores every later request. The Rust
//! closure only writes a frame and waits; it does not call CUDA.

use std::{
    panic::{UnwindSafe, catch_unwind, resume_unwind},
    process::{Child, Command, Stdio},
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result, bail};

use crate::frame::{self, WorkerResponse};

struct Pipes {
    stdin: std::process::ChildStdin,
    stdout: std::process::ChildStdout,
}

pub(crate) struct Executor {
    child: Mutex<Child>,
    pipes: Mutex<Pipes>,
    ready: AtomicBool,
    health: Vec<u8>,
}

impl Executor {
    pub(crate) fn spawn(mut command: Command) -> Result<Self> {
        command.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = command.spawn().context("spawning the Clef executor")?;
        let stdin = child.stdin.take().context("executor stdin")?;
        let mut stdout = child.stdout.take().context("executor stdout")?;
        let started = frame::read_response(&mut stdout);
        let health = match started {
            Ok(response) if response.status == 200 => response.body,
            other => {
                let status = reap(&mut child);
                match other {
                    Ok(response) => bail!(
                        "executor startup status {} ({status}); the port was not opened",
                        response.status
                    ),
                    Err(error) => bail!(
                        "executor exited during startup ({status}): {error:#}; the port was not opened"
                    ),
                }
            }
        };
        Ok(Self {
            child: Mutex::new(child),
            pipes: Mutex::new(Pipes { stdin, stdout }),
            ready: AtomicBool::new(true),
            health,
        })
    }

    pub(crate) fn ready(&self) -> bool {
        if !self.ready.load(Ordering::Acquire) {
            return false;
        }
        // Health must notice an idle process exit without sending an inference.
        let alive = self
            .child
            .lock()
            .is_ok_and(|mut child| matches!(child.try_wait(), Ok(None)));
        if !alive {
            self.ready.store(false, Ordering::Release);
        }
        alive
    }

    pub(crate) fn health(&self) -> &[u8] {
        &self.health
    }

    pub(crate) fn exchange(&self, body: &[u8]) -> Result<WorkerResponse> {
        self.with_pipes(|pipes| {
            frame::write_request(&mut pipes.stdin, frame::OPCODE_DECIDE, body)?;
            frame::read_response(&mut pipes.stdout)
        })
    }

    fn with_pipes(
        &self,
        exchange: impl FnOnce(&mut Pipes) -> Result<WorkerResponse> + UnwindSafe,
    ) -> Result<WorkerResponse> {
        if !self.ready() {
            bail!("executor unavailable");
        }
        let result = catch_unwind(|| {
            let mut pipes = self
                .pipes
                .lock()
                .map_err(|_| anyhow::anyhow!("executor lock poisoned"))?;
            exchange(&mut pipes)
        });
        if !matches!(result, Ok(Ok(_))) {
            // Publish errors and panics before releasing admission, even if the
            // caller disconnected. Unwinding has already released the pipes lock.
            self.fail();
        }
        match result {
            Ok(result) => result,
            Err(panic) => resume_unwind(panic),
        }
    }

    pub(crate) fn fail(&self) {
        self.ready.store(false, Ordering::Release);
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn reap(child: &mut Child) -> String {
    let _ = child.kill();
    match child.wait() {
        Ok(status) => status.to_string(),
        Err(error) => error.to_string(),
    }
}

#[cfg(test)]
#[path = "../../../../../tests/clef/executor.rs"]
mod tests;
