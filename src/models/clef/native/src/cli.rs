//! Arguments for `omni-clef`. The executor process receives the checkpoint
//! flags. Host and port stay in this process.

use std::{
    env,
    ffi::OsStr,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use tokio::{net::TcpListener, signal};

use crate::server::{Engine, router};

pub enum Cli {
    Help,
    Serve(ServeArgs),
}

pub struct ServeArgs {
    pub model: PathBuf,
    pub variant: String,
    pub device: String,
    pub gdn_prefill: String,
    pub host: String,
    pub port: u16,
}

const HELP: &str = "\
omni-clef — Rust HTTP server for the pinned Clef executor

The executor is one Python process. Its main thread loads CUDA and scores
every request. This process admits one request at a time.

Usage:
  omni-clef --model <snapshot> [options]

Options:
  --model <dir>          Snapshot directory of the pinned revision
  --variant <name>       clef-flash (default) or clef
  --device <device>      cuda (default)
  --gdn-prefill <name>   fla (default) or flashinfer
  --host <addr>          127.0.0.1 (default)
  --port <port>          8000 (default)
  --help                 Show this help

Environment:
  CLEF_PYTHON            Interpreter for the executor. Default: python3.
                         Point this at the recipe venv. That venv has
                         causal-conv1d and, for --gdn-prefill flashinfer,
                         flashinfer-python==0.6.16.post3. System python3
                         does not, and the executor exits before the port
                         opens. fla stays the default until answer parity
                         passes.
  CLEF_PYTHONPATH        Directory that contains models/clef/worker.py.
";

pub fn parse_cli<I, T>(args: I) -> Result<Cli>
where
    I: IntoIterator<Item = T>,
    T: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(Cli::Help);
    }
    let mut model = None;
    let mut variant = "clef-flash".to_string();
    let mut device = "cuda".to_string();
    let mut gdn_prefill = "fla".to_string();
    let mut host = "127.0.0.1".to_string();
    let mut port = 8000u16;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].clone();
        let Some(value) = args.get(index + 1) else {
            bail!("missing value for {flag}");
        };
        let value = value.clone();
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(&value)),
            "--variant" => variant = value,
            "--device" => device = value,
            "--gdn-prefill" => gdn_prefill = value,
            "--host" => host = value,
            "--port" => {
                port = value
                    .parse()
                    .with_context(|| format!("invalid port {value}"))?
            }
            other if other.starts_with('-') => bail!("unknown argument {other}"),
            other => bail!("unexpected argument {other}"),
        }
        index += 2;
    }
    let Some(model) = model else {
        bail!("--model is required");
    };
    if variant != "clef" && variant != "clef-flash" {
        bail!("variant must be clef or clef-flash");
    }
    if gdn_prefill != "fla" && gdn_prefill != "flashinfer" {
        bail!("gdn-prefill must be fla or flashinfer");
    }
    if host.is_empty() {
        bail!("host must not be empty");
    }
    Ok(Cli::Serve(ServeArgs {
        model,
        variant,
        device,
        gdn_prefill,
        host,
        port,
    }))
}

pub fn locate_pythonpath_from(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    for _ in 0..8 {
        let candidate = dir.join("src");
        if candidate.join("models/clef/worker.py").is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

pub fn locate_pythonpath() -> Result<PathBuf> {
    if let Some(path) = env::var_os("CLEF_PYTHONPATH") {
        let path = PathBuf::from(path);
        if path.join("models/clef/worker.py").is_file() {
            return Ok(path);
        }
        bail!("CLEF_PYTHONPATH does not contain models/clef/worker.py");
    }
    if let Ok(cwd) = env::current_dir()
        && let Some(path) = locate_pythonpath_from(&cwd)
    {
        return Ok(path);
    }
    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
        && let Some(path) = locate_pythonpath_from(parent)
    {
        return Ok(path);
    }
    bail!(
        "cannot find src/models/clef/worker.py; run from the repository root or set CLEF_PYTHONPATH"
    )
}

pub fn executor_python_from(value: Option<std::ffi::OsString>) -> std::ffi::OsString {
    value.unwrap_or_else(|| std::ffi::OsString::from("python3"))
}

pub fn python_worker_command(args: &ServeArgs, python: &OsStr, pythonpath: &Path) -> Command {
    let mut command = Command::new(python);
    command.arg("-u");
    command.arg("-m");
    command.arg("models.clef.worker");
    command.arg("--model");
    command.arg(&args.model);
    command.arg("--variant");
    command.arg(&args.variant);
    command.arg("--device");
    command.arg(&args.device);
    command.arg("--gdn-prefill");
    command.arg(&args.gdn_prefill);
    command.env("PYTHONUNBUFFERED", "1");
    command.stderr(Stdio::inherit());
    let mut path = pythonpath.as_os_str().to_os_string();
    if let Some(existing) = env::var_os("PYTHONPATH") {
        path.push(":");
        path.push(existing);
    }
    command.env("PYTHONPATH", path);
    command
}

pub fn print_help() {
    print!("{HELP}");
}

pub async fn serve_from_env() -> Result<()> {
    match parse_cli(env::args().skip(1))? {
        Cli::Help => {
            print_help();
            Ok(())
        }
        Cli::Serve(args) => {
            let python = executor_python_from(env::var_os("CLEF_PYTHON"));
            let pythonpath = locate_pythonpath()?;
            let command = python_worker_command(&args, &python, &pythonpath);
            let engine = Arc::new(Engine::spawn(command)?);
            let listener = TcpListener::bind((args.host.as_str(), args.port))
                .await
                .with_context(|| format!("binding {}:{}", args.host, args.port))?;
            let addr: SocketAddr = listener.local_addr()?;
            eprintln!("omni-clef listening on {addr}");
            axum::serve(listener, router(engine))
                .with_graceful_shutdown(shutdown_signal())
                .await?;
            Ok(())
        }
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut sigterm) => {
                sigterm.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = signal::ctrl_c() => {}
        () = terminate => {}
    }
}
