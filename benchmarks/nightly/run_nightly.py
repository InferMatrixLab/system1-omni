#!/usr/bin/env python3
"""Local nightly GPU benchmark for hk2.

Operators install ``benchmarks/nightly/run_nightly.sh`` from cron or systemd
on a Linux GPU host. The job pulls a configured git ref, attaches to or starts
``POST /v1/systemone``, and runs the existing ``benchmarks/bench.py`` client.

Default arms are Open-Jev-9B and Open-Jev-27B-v1.1, Public231 text, concurrency
1. Intern-Decision 0.8B / 2B / 4B are opt-in. C8/C16 and the packing probe are
not headline numbers. GitHub-hosted runners do not execute this suite.

Missing Open-Jev merged weights (``download`` + ``export_merged``) exit
immediately with a Chinese and English reason.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "system1-omni-nightly/1"
CONFIG_SCHEMA = "system1-omni-nightly-config/1"
EXPORT_FORMAT = "open-jev-text-merged/1"
EXPORT_NAME = "open_jev_export.json"
LATENCY_BASIS = "post_to_body_json_and_schema_validation"
QUANTILE = "p50=statistics.median; p95=nearest-rank ceil(0.95*n)-1"
CLOCK_NOTE = (
    "bench.py clock includes body receipt, JSON parsing and schema validation. "
    "Do not compare these files with the historical H800 body-receipt board."
)
PACKING_NOTE = (
    "Cheap multi-candidate C1 probe on Open-Jev-27B-v1.1 only. "
    "Not the H200 prefill_batching campaign and not a headline number."
)
REGRESSION_NOTE = (
    "Regression gate only. Not a latency-leaderboard or war-report number."
)
OPTIONAL_NOTE = (
    "Opt-in official HF arm. Not required every night. "
    "Shown on the five-model board only when this run enabled it."
)
SMOKE_NOTE = "Contract smoke. Not a Public231 headline number."
CUSTOM_NOTE = "Custom manifest. Not a Public231 headline number."
PUBLIC231_REQUESTS = 231
SHANGHAI = ZoneInfo("Asia/Shanghai")
# Search order for an out-of-tree wrapper. run_nightly.sh must match.
CONVENTIONAL_REPOS = (
    "repos/ThinkFlowLab/system1-omni",
    "repos/InferMatrixLab/system1-omni",
)
# Serving and weights for these ids are not in the nightly set.
EXCLUDED_PREFIXES = (
    "d3",
    "vela",
    "valen",
    "decider",
    "laya",
    "jemm",
    "cua",
    "lfm",
    "clm",
    "omnijev",
    "jev-vl",
)
REPORT_KEYS = (
    "schema",
    "illustrative",
    "note",
    "date",
    "timezone",
    "started_at",
    "finished_at",
    "host",
    "operator",
    "git_sha",
    "git_ref",
    "git_dirty",
    "gpu",
    "gpu_ids",
    "driver",
    "cuda",
    "compute_cap",
    "precision",
    "latency_basis",
    "quantile",
    "comparable_to_h800_historical_board",
    "clock_note",
    "workload",
    "manifest_sha256",
    "manifest_requests",
    "headline_p50_ms",
    "headline_p95_ms",
    "arms",
    "packing_probe",
    "ok",
    "failures",
    "issue",
)
METADATA_FIELDS = (
    "gpu",
    "gpu_ids",
    "driver",
    "cuda",
    "precision",
    "model_revision",
    "runtime_revision",
    "cache_policy",
    "cuda_evidence",
    "reservation",
)


class NightlyError(Exception):
    """A bilingual, non-hanging nightly failure."""

    def __init__(self, english, chinese, details="", code=2):
        self.english = english
        self.chinese = chinese
        self.details = details
        self.code = code
        super().__init__(str(self))

    def __str__(self):
        lines = [f"error: {self.english}", f"错误：{self.chinese}"]
        if self.details:
            lines.append(str(self.details).rstrip())
        return "\n".join(lines) + "\n"


def redact(text):
    return re.sub(r"x-access-token:[^@\s]+@", "x-access-token:[REDACTED]@", text or "")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flag(cli_value, env_name, default):
    if cli_value is not None:
        return cli_value
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def positive_int(value, name, upper):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise NightlyError(
            f"{name} must be an integer",
            f"{name} 必须是整数",
        ) from exc
    if not 1 <= number <= upper:
        raise NightlyError(
            f"{name} must be within 1..{upper}",
            f"{name} 必须在 1 到 {upper} 之间",
        )
    return number


def load_config(path=None):
    config_path = Path(path) if path else HERE / "config.example.json"
    if not config_path.is_file():
        raise NightlyError(
            f"nightly config not found: {config_path}",
            f"找不到 nightly 配置：{config_path}",
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NightlyError(
            f"nightly config is unreadable: {config_path}",
            f"nightly 配置无法读取：{config_path}",
            str(exc),
        ) from exc
    if config.get("schema") != CONFIG_SCHEMA or not config.get("models"):
        raise NightlyError(
            "nightly config schema or models list is missing",
            "nightly 配置缺少 schema 或 models",
        )
    return config


def assert_allowed(model):
    key = str(model.get("id", "")).lower().replace("_", "-")
    for prefix in EXCLUDED_PREFIXES:
        if key == prefix or key.startswith(prefix + "-"):
            raise NightlyError(
                f"{model.get('id')} is outside the nightly set until its serving and weights have merged.",
                f"{model.get('id')} 尚未纳入 nightly：需先合并 serving 与权重。",
            )
    if model.get("launch") not in {"open-jev-native", "attach-only"}:
        raise NightlyError(
            f"{model.get('id')} has an unsupported launch mode",
            f"{model.get('id')} 的启动方式不受支持",
        )
    if model.get("cadence") not in {"nightly", "optional"}:
        raise NightlyError(
            f"{model.get('id')} has an unsupported cadence",
            f"{model.get('id')} 的 cadence 不受支持",
        )


def selected_models(config, include_intern):
    chosen = []
    for model in config["models"]:
        assert_allowed(model)
        if model["cadence"] == "nightly":
            chosen.append(model)
        elif include_intern:
            chosen.append(model)
    if not chosen:
        raise NightlyError("no nightly models are configured", "没有配置 nightly 模型")
    return chosen


def select_conventional_repo(home, exists):
    """First conventional checkout that contains this harness."""
    for relative in CONVENTIONAL_REPOS:
        path = Path(home) / relative
        if exists(path / "benchmarks" / "nightly" / "run_nightly.py"):
            return path
    return None


def check_endpoint(url):
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise NightlyError(
            f"endpoint must be an http(s) URL: {url}",
            f"endpoint 必须是 http(s) URL：{url}",
        )
    if parts.username or parts.password:
        raise NightlyError(
            "endpoint must not contain credentials",
            "endpoint 不能包含账号或密码",
        )
    if parts.query or parts.fragment:
        raise NightlyError(
            "endpoint must not contain a query or fragment",
            "endpoint 不能包含 query 或 fragment",
        )
    if not parts.path.rstrip("/").endswith("/v1/systemone"):
        raise NightlyError(
            "endpoint path must end with /v1/systemone",
            "endpoint 路径必须以 /v1/systemone 结尾",
        )
    return url.strip().rstrip("/")


def origin_health(endpoint):
    parts = urlsplit(endpoint)
    return f"{parts.scheme}://{parts.netloc}/health"


def classify_workload(path, smoke):
    if smoke or path.name == "smoke.jsonl":
        return "smoke"
    if "public231" in path.name.lower():
        return "public231"
    return "custom"


def resolve_manifest(args, config, repo):
    if args.smoke:
        path = repo / "benchmarks" / "smoke.jsonl"
        return path, "smoke"
    if args.manifest:
        raw = args.manifest
    elif os.environ.get("OMNI_NIGHTLY_MANIFEST"):
        raw = os.environ["OMNI_NIGHTLY_MANIFEST"]
    else:
        raw = config.get("default_manifest") or ""
    if not raw:
        raise NightlyError(
            "Public231 manifest is not configured",
            "没有配置 Public231 清单",
        )
    path = Path(os.path.expanduser(raw))
    if not path.is_absolute():
        path = repo / path
    workload = (
        args.workload
        or os.environ.get("OMNI_NIGHTLY_WORKLOAD")
        or classify_workload(path, False)
    )
    if workload not in {"public231", "smoke", "custom"}:
        raise NightlyError(
            "workload must be public231, smoke or custom",
            "workload 只能是 public231、smoke 或 custom",
        )
    return path, workload


def load_manifest(path, workload):
    if not path.is_file():
        raise NightlyError(
            "Benchmark manifest is missing. The headline job expects the frozen Public231 JSONL. Pass --smoke only for a contract check.",
            "找不到请求清单。Headline 需要冻结的 Public231 JSONL。只有契约检查才使用 --smoke。",
            str(path),
        )
    bench = load_bench()
    try:
        cases = bench.load_cases(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise NightlyError(
            "manifest failed bench.py validation",
            "清单未通过 bench.py 校验",
            str(exc),
        ) from exc
    if workload == "public231" and len(cases) != PUBLIC231_REQUESTS:
        raise NightlyError(
            f"Public231 manifest has {len(cases)} requests; expected {PUBLIC231_REQUESTS}.",
            f"Public231 清单有 {len(cases)} 条请求，应为 {PUBLIC231_REQUESTS}。拒绝把它当作 headline。",
            str(path),
        )
    pinned = os.environ.get("OMNI_NIGHTLY_MANIFEST_SHA256", "").strip()
    file_sha = digest(path)
    if pinned and pinned != file_sha:
        raise NightlyError(
            "manifest SHA256 does not match OMNI_NIGHTLY_MANIFEST_SHA256",
            "清单 SHA256 与 OMNI_NIGHTLY_MANIFEST_SHA256 不一致",
            f"expected {pinned}\nactual   {file_sha}",
        )
    return cases, file_sha


def load_bench():
    try:
        from benchmarks import bench
    except ModuleNotFoundError as exc:
        raise NightlyError(
            "install benchmarks/requirements.txt (httpx) before the nightly job.",
            "请先安装 benchmarks/requirements.txt（需要 httpx），再运行 nightly。",
            str(exc),
        ) from exc
    return bench


def check_merged_export(repo, model):
    """Return an error string when download + export_merged is incomplete."""
    path = repo / model["weights"]
    marker = path / EXPORT_NAME
    if not marker.is_file():
        return (
            f"missing {marker}. download + export_merged is not done "
            f"for {model['display']}."
        )
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable {marker}: {exc}"
    if payload.get("format") != EXPORT_FORMAT:
        return (
            f"{marker} is not {EXPORT_FORMAT}. "
            "A plain download is not a merged export; run export_merged."
        )
    expected = {
        "model_id": model["export_model_id"],
        "base_revision": model["base_revision"],
        "checkpoint_revision": model["checkpoint_revision"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return f"{marker} {key} is {payload.get(key)!r}; expected {value!r}. Re-run export_merged."
    shards = [
        item
        for item in path.glob("*.safetensors")
        if item.is_file() and item.stat().st_size > 0
    ]
    if not shards:
        return f"{path} has no safetensors shards. export_merged did not finish."
    return None


def require_weights(repo, models):
    errors = []
    for model in models:
        if model["launch"] != "open-jev-native":
            continue
        if os.environ.get(model["endpoint_env"], "").strip():
            continue
        found = check_merged_export(repo, model)
        if found:
            errors.append(found)
    if errors:
        raise NightlyError(
            "Open-Jev merged weights are missing or incomplete. Run download and export_merged before the nightly job. See recipe/open_jev/native.md.",
            "缺少或不完整的 Open-Jev 合并权重。请先完成 download 与 export_merged，再运行 nightly。参见 recipe/open_jev/native.md。",
            "\n".join(errors),
        )


def launching_models(models):
    return [
        model
        for model in models
        if model["launch"] == "open-jev-native"
        and not os.environ.get(model["endpoint_env"], "").strip()
    ]


def require_binaries(repo, models):
    if not launching_models(models):
        return
    missing = [
        name
        for name in ("omni-open-jev-native", "omni-jev")
        if not (repo / "target" / "release" / name).is_file()
    ]
    if missing:
        raise NightlyError(
            "release binaries are missing ("
            + ", ".join(missing)
            + "). Build them with the Open-Jev native recipe or set OMNI_NIGHTLY_BUILD=1.",
            "缺少 release 二进制（"
            + "、".join(missing)
            + "）。请按 Open-Jev native 配方构建，或设置 OMNI_NIGHTLY_BUILD=1。",
            "See recipe/open_jev/native.md.",
        )


def git(repo, *args, timeout):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NightlyError(
            f"git {' '.join(args)} timed out after {timeout}s",
            f"git {' '.join(args)} 超时（{timeout} 秒），已停止。",
        ) from exc
    except OSError as exc:
        raise NightlyError(
            f"git failed to start: {exc}",
            f"无法启动 git：{exc}",
        ) from exc
    if result.returncode != 0:
        detail = redact((result.stderr or result.stdout or "").strip())
        raise NightlyError(
            f"git {' '.join(args)} failed",
            f"git {' '.join(args)} 失败",
            detail,
        )
    return result.stdout.strip()


def validate_ref(ref):
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", ref) or ".." in ref.split("/"):
        raise NightlyError(
            f"refusing git ref {ref!r}",
            f"拒绝 git ref {ref!r}",
        )
    return ref


def sync_checkout(repo, ref):
    validate_ref(ref)
    dirty = git(repo, "status", "--porcelain", timeout=30)
    if dirty:
        raise NightlyError(
            "nightly checkout is dirty; refusing to pull. Use a dedicated clone.",
            "nightly 工作区不干净，已拒绝 pull。请使用专用克隆。",
            dirty,
        )
    git(repo, "fetch", "--prune", "origin", ref, timeout=180)
    git(repo, "checkout", ref, timeout=60)
    git(repo, "pull", "--ff-only", "origin", ref, timeout=180)
    return git(repo, "rev-parse", "HEAD", timeout=30)


def read_provenance(repo, required):
    try:
        sha = git(repo, "rev-parse", "HEAD", timeout=30)
        dirty = bool(git(repo, "status", "--porcelain", timeout=30))
    except NightlyError:
        if required:
            raise
        return "unknown", True
    return sha, dirty


def maybe_update(repo, ref, enabled):
    if not enabled or os.environ.get("OMNI_NIGHTLY_REEXEC") == "1":
        return read_provenance(repo, required=enabled)
    before = git(repo, "rev-parse", "HEAD", timeout=30)
    after = sync_checkout(repo, ref)
    if before != after:
        os.environ["OMNI_NIGHTLY_REEXEC"] = "1"
        script = str(Path(__file__).resolve())
        argv = [sys.executable, script, *sys.argv[1:]]
        os.execv(sys.executable, argv)
    return after, False


def percentiles(samples):
    if not samples:
        return None, None
    ordered = sorted(samples)
    index = math.ceil(0.95 * len(ordered)) - 1
    return statistics.median(ordered), ordered[index]


def parse_concurrency_list(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return []
    items = []
    for part in text.split(","):
        try:
            item = int(part.strip())
        except ValueError as exc:
            raise NightlyError(
                f"invalid regression concurrency {part!r}",
                f"回归并发无效：{part!r}",
            ) from exc
        if item < 2:
            raise NightlyError(
                "regression concurrency must be >= 2; C1 is the headline arm.",
                "回归并发必须 >= 2；C1 是 headline，不是回归门禁。",
            )
        items.append(item)
    if len(items) != len(set(items)):
        raise NightlyError("duplicate regression concurrency", "回归并发重复。")
    return items


def visible_devices():
    raw = os.environ.get(
        "OMNI_NIGHTLY_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    )
    return raw.strip() or "0"


def gpu_memory_used_mib(index):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    token = result.stdout.strip().splitlines()[0].split(",")[0].strip()
    try:
        return int(token)
    except ValueError:
        return None


def require_gpu_idle(models):
    if not launching_models(models):
        return
    raw_limit = os.environ.get("OMNI_NIGHTLY_MAX_GPU_USED_MIB", "2048").strip()
    if raw_limit in {"", "-1"}:
        return
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise NightlyError(
            "OMNI_NIGHTLY_MAX_GPU_USED_MIB must be an integer or -1",
            "OMNI_NIGHTLY_MAX_GPU_USED_MIB 必须是整数或 -1",
        ) from exc
    if limit < 0:
        return
    index = visible_devices().split(",")[0].strip()
    used = gpu_memory_used_mib(index)
    if used is None or used <= limit:
        return
    raise NightlyError(
        f"GPU {index} already has {used} MiB in use (limit {limit}). Refusing to start another worker.",
        f"GPU {index} 已占用 {used} MiB（上限 {limit}）。拒绝再启动 worker，没有继续等待。",
    )


def query_gpu(visible):
    info = {
        "gpu": "unavailable",
        "gpu_ids": visible or "unavailable",
        "driver": "unavailable",
        "cuda": "unavailable",
        "compute_cap": "unavailable",
    }
    try:
        listed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return info
    if listed.returncode != 0:
        return info
    rows = []
    for line in listed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(parts)
    if not rows:
        return info
    wanted = {part.strip() for part in visible.split(",") if part.strip()}
    chosen = [row for row in rows if row[0] in wanted or row[2] in wanted] or [rows[0]]
    info["gpu"] = chosen[0][1]
    info["gpu_ids"] = " ".join(row[2] for row in chosen)
    info["driver"] = chosen[0][3]
    info["compute_cap"] = chosen[0][4]
    try:
        header = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return info
    match = re.search(r"CUDA Version:\s*([0-9.]+)", header.stdout or "")
    if match:
        info["cuda"] = match.group(1)
    return info


def metadata_for(model, repo, provenance, gpu_info, host):
    if model["launch"] == "open-jev-native":
        revision = (
            f"{model['export_model_id']} base {model['base_revision']} "
            f"checkpoint {model['checkpoint_revision']}"
        )
        precision_source = "declared from the Open-Jev native recipe (BF16 merged export); not probed at runtime"
        cache_policy = (
            "CUA_S1_GRAPH unset (eager). Open-Jev-27B-v1.1 packs candidates inside one request; "
            "Open-Jev-9B does not. Prefix cache off."
        )
    else:
        revision = model.get("hub_id") or model["id"]
        precision_source = (
            "declared BF16 to match the Public231 campaign; this harness does not inspect the official HF process. "
            "score_decimals stays at the bench.py default unless the model config overrides it. "
            "Native Decider in this repo is a different server and uses two decimals."
        )
        cache_policy = (
            "official HF server; cache policy is whatever that process already uses"
        )
    payload = {
        "gpu": gpu_info["gpu"],
        "gpu_ids": gpu_info["gpu_ids"],
        "driver": gpu_info["driver"],
        "cuda": gpu_info["cuda"],
        "precision": model.get("precision") or "bf16",
        "precision_source": precision_source,
        "model_revision": revision,
        "runtime_revision": provenance,
        "cache_policy": cache_policy,
        "cuda_evidence": (
            "HTTP /health and POST /v1/systemone via bench.py. "
            "This client does not prove which kernels ran."
        ),
        "reservation": (
            f"local nightly on {host}; CUDA_VISIBLE_DEVICES={visible_devices()}; "
            "no Slurm reservation recorded by this harness"
        ),
        "host": host,
    }
    missing = [name for name in METADATA_FIELDS if not payload.get(name)]
    if missing:
        raise NightlyError(
            "benchmark metadata is incomplete: " + ", ".join(missing),
            "benchmark metadata 不完整：" + "、".join(missing),
        )
    return payload


def launch_spec(model, repo, visible, cuda_lib=None):
    worker = repo / "target" / "release" / "omni-open-jev-native"
    frontend = repo / "target" / "release" / "omni-jev"
    weights = repo / model["weights"]
    worker_env = os.environ.copy()
    worker_env.pop("CUA_S1_GRAPH", None)
    worker_env["OPEN_JEV_MODEL"] = str(weights)
    worker_env["OPEN_JEV_HOST"] = "127.0.0.1"
    worker_env["OPEN_JEV_PORT"] = str(model["worker_port"])
    worker_env["CUDA_VISIBLE_DEVICES"] = visible
    if cuda_lib:
        worker_env["OPEN_JEV_CUDA_LIB"] = cuda_lib
    elif os.environ.get("OPEN_JEV_CUDA_LIB"):
        worker_env["OPEN_JEV_CUDA_LIB"] = os.environ["OPEN_JEV_CUDA_LIB"]
    front_env = os.environ.copy()
    front_env["OMNI_JEV_BIND"] = f"127.0.0.1:{model['frontend_port']}"
    front_env["OMNI_JEV_BACKEND_URL"] = f"http://127.0.0.1:{model['worker_port']}"
    return {
        "worker_argv": [str(worker)],
        "worker_env": worker_env,
        "frontend_argv": [str(frontend)],
        "frontend_env": front_env,
        "worker_health": f"http://127.0.0.1:{model['worker_port']}/health",
        "endpoint": f"http://127.0.0.1:{model['frontend_port']}/v1/systemone",
    }


def public_launch(spec):
    return {
        "worker_argv": spec["worker_argv"],
        "frontend_argv": spec["frontend_argv"],
        "open_jev_model": spec["worker_env"]["OPEN_JEV_MODEL"],
        "cuda_visible_devices": spec["worker_env"]["CUDA_VISIBLE_DEVICES"],
        "cuda_graph": "CUA_S1_GRAPH" in spec["worker_env"],
        "endpoint": spec["endpoint"],
    }


def probe_health(endpoint, timeout=2.0):
    import httpx

    try:
        response = httpx.get(
            origin_health(endpoint),
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return {
            "ok": False,
            "status": response.status_code,
            "body": response.text[:200],
        }
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return {
            "ok": False,
            "status": response.status_code,
            "body": response.text[:200],
        }
    if not isinstance(payload, dict):
        return {"ok": False, "status": response.status_code}
    found = dict(payload)
    found["ok"] = True
    return found


def plan_service(model, repo, env, probe):
    configured = env.get(model["endpoint_env"], "").strip()
    if model["launch"] == "attach-only" and not configured:
        raise NightlyError(
            f"{model['display']} is opt-in and needs {model['endpoint_env']}. The harness does not start official HF servers.",
            f"{model['display']} 为可选模型，需要环境变量 {model['endpoint_env']}。本程序不启动官方 HF 服务。",
        )
    if configured:
        endpoint = check_endpoint(configured)
        health = probe(endpoint)
        if not health or not health.get("ok"):
            raise NightlyError(
                f"{model['id']} endpoint is down: {endpoint}",
                f"{model['id']} 的 endpoint 无响应：{endpoint}。不会等待挂起。",
            )
        expect = model.get("export_model_id")
        if expect and health.get("model") != expect:
            raise NightlyError(
                f"health model {health.get('model')!r} does not match {expect}",
                f"health 模型与 {expect} 不符，拒绝附着。",
            )
        return {"mode": "attach", "endpoint": endpoint, "health": health}
    if model["launch"] != "open-jev-native":
        raise NightlyError(
            f"{model['id']} cannot be launched by this harness",
            f"本 harness 不能启动 {model['id']}",
        )
    spec = launch_spec(model, repo, visible_devices())
    health = probe(spec["endpoint"])
    if health and health.get("ok") and health.get("model") == model["export_model_id"]:
        return {"mode": "attach", "endpoint": spec["endpoint"], "health": health}
    if health and health.get("ok"):
        raise NightlyError(
            f"frontend port {model['frontend_port']} already serves {health.get('model')}",
            f"端口 {model['frontend_port']} 已有其他模型，拒绝抢占。",
        )
    return {"mode": "launch", "endpoint": spec["endpoint"], "spec": spec}


class ManagedServer:
    def __init__(self, repo):
        self.repo = repo
        self.procs = []
        self.logs = []

    def spawn(self, argv, env, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        self.logs.append(handle)
        proc = subprocess.Popen(
            argv,
            cwd=self.repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.procs.append((proc, log_path))
        return proc

    def close(self):
        import signal

        for proc, _log in self.procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        deadline = time.monotonic() + 15
        for proc, _log in self.procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait(timeout=5)
        for handle in self.logs:
            handle.close()
        self.procs.clear()
        self.logs.clear()


def tail(path, limit=40):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])[-2000:]


def wait_ready(endpoint, expected_model, proc, log_path, timeout):
    deadline = time.monotonic() + timeout
    last = "no response"
    while True:
        if proc is not None and proc.poll() is not None:
            raise NightlyError(
                f"server exited {proc.returncode} before /health was ready",
                f"服务在 /health 就绪前退出（码 {proc.returncode}）。没有继续等待。",
                tail(log_path),
                code=1,
            )
        health = probe_health(endpoint, timeout=2.0)
        if (
            health
            and health.get("ok")
            and (expected_model is None or health.get("model") == expected_model)
        ):
            return health
        if (
            health
            and health.get("ok")
            and expected_model
            and health.get("model") not in (None, expected_model)
        ):
            raise NightlyError(
                f"health model {health.get('model')!r} does not match {expected_model}",
                f"health 模型与 {expected_model} 不符。",
                code=1,
            )
        if health:
            last = json.dumps(health, ensure_ascii=False)[:300]
        if time.monotonic() >= deadline:
            raise NightlyError(
                f"timed out after {timeout}s waiting for /health ({expected_model}). Last: {last}",
                f"等待 /health（模型 {expected_model}）超时（{timeout} 秒），已停止，没有挂起。最近响应：{last}",
                tail(log_path),
                code=1,
            )
        time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))


def start_or_attach(model, repo, log_dir, ready_timeout, probe=probe_health):
    plan = plan_service(model, repo, os.environ, probe)
    if plan["mode"] == "attach":
        return plan["endpoint"], True, ManagedServer(repo)
    server = ManagedServer(repo)
    spec = plan["spec"]
    try:
        worker_log = log_dir / f"{model['id']}-worker.log"
        worker = server.spawn(spec["worker_argv"], spec["worker_env"], worker_log)
        wait_ready(
            spec["worker_health"],
            model["export_model_id"],
            worker,
            worker_log,
            ready_timeout,
        )
        front_log = log_dir / f"{model['id']}-frontend.log"
        frontend = server.spawn(spec["frontend_argv"], spec["frontend_env"], front_log)
        wait_ready(spec["endpoint"], model["export_model_id"], frontend, front_log, 30)
        return spec["endpoint"], False, server
    except BaseException:
        server.close()
        raise


def bench_budget(case_count, per_request, warmup):
    """Worst case of every request hitting its HTTP timeout, capped at two hours."""
    worst = int(per_request) * (int(case_count) + int(warmup)) + 120
    return min(worst, 7200)


def run_bench(
    repo,
    manifest,
    endpoint,
    model,
    phase,
    concurrency,
    warmup,
    timeout,
    metadata_path,
    output,
    score_decimals,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise NightlyError(
            f"bench output already exists: {output}",
            f"bench 输出目录已存在：{output}",
            code=1,
        )
    command = [
        sys.executable,
        str(repo / "benchmarks" / "bench.py"),
        "run",
        str(manifest),
        "--endpoint",
        endpoint,
        "--model",
        model["alias"],
        "--protocol",
        "systemone",
        "--score-decimals",
        str(score_decimals),
        "--metadata",
        str(metadata_path),
        "--phase",
        phase,
        "--concurrency",
        str(concurrency),
        "--warmup",
        str(warmup),
        "--timeout",
        str(timeout),
        "--output",
        str(output),
    ]
    bench = load_bench()
    case_count = len(bench.load_cases(manifest))
    budget = bench_budget(case_count, timeout, warmup)
    log_path = output.parent / f"{output.name}.log"
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            timeout=budget,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NightlyError(
            f"bench.py exceeded {budget}s and was killed",
            f"bench.py 超过 {budget} 秒，已终止，没有继续等待。",
            redact(str(exc)),
            code=1,
        ) from exc
    log_path.write_text(
        redact(completed.stdout or "") + "\n" + redact(completed.stderr or ""),
        encoding="utf-8",
    )
    summary_path = output / "summary.json"
    summary = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = None
    ok = (
        completed.returncode == 0
        and isinstance(summary, dict)
        and summary.get("evidence_complete") is True
        and summary.get("failed_requests") == 0
    )
    return {
        "returncode": completed.returncode,
        "summary": summary,
        "output": str(output),
        "ok": ok,
        "command": command,
    }


def collect_samples(output):
    path = Path(output) / "responses.jsonl"
    if not path.is_file():
        return []
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        latency = record.get("latency_ms")
        if (
            record.get("state") == "completed"
            and record.get("error") is None
            and isinstance(latency, (int, float))
            and not isinstance(latency, bool)
        ):
            samples.append(float(latency))
    return samples


def pool_runs(runs):
    samples = []
    planned = successful = failed = incomplete = unattempted = 0
    evidence = bool(runs)
    rounds = []
    for run in runs:
        summary = run.get("summary") or {}
        evidence = evidence and run.get("ok") is True
        planned += summary.get("planned_requests") or 0
        successful += summary.get("successful_requests") or 0
        failed += summary.get("failed_requests") or 0
        incomplete += summary.get("incomplete_requests") or 0
        unattempted += summary.get("unattempted_requests") or 0
        samples.extend(collect_samples(run["output"]))
        rounds.append(
            {
                "output": run["output"],
                "returncode": run["returncode"],
                "ok": run["ok"],
                "p50_ms": summary.get("successful_latency_p50_ms"),
                "p95_ms": summary.get("successful_latency_p95_ms"),
                "successful_requests": summary.get("successful_requests"),
                "failed_requests": summary.get("failed_requests"),
                "evidence_complete": summary.get("evidence_complete"),
            }
        )
    p50, p95 = percentiles(samples)
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "sample_count": len(samples),
        "planned_requests": planned,
        "successful_requests": successful,
        "failed_requests": failed,
        "incomplete_requests": incomplete,
        "unattempted_requests": unattempted,
        "evidence_complete": evidence,
        "rounds": rounds,
    }


def arm_record(
    model, pooled, workload, concurrency, role, endpoint, attached, manifest_sha, note
):
    headline = role == "headline" and workload == "public231" and concurrency == 1
    eligible = (
        (
            headline
            or (role == "optional" and workload == "public231" and concurrency == 1)
        )
        and pooled["evidence_complete"]
        and pooled["failed_requests"] == 0
        and pooled["sample_count"] > 0
    )
    if role != "headline":
        headline = False
    return {
        "id": model["id"],
        "display": model["display"],
        "role": role,
        "headline": headline and eligible,
        "leaderboard_eligible": eligible,
        "workload": workload,
        "concurrency": concurrency,
        "phase": "measured",
        "manifest_sha256": manifest_sha,
        "endpoint": endpoint,
        "model_alias": model["alias"],
        "attached": attached,
        "score_decimals": model.get("score_decimals", 4),
        "hub_id": model.get("hub_id"),
        "note": note,
        **pooled,
    }


def previous_packing(results_dir, manifest_sha):
    latest = results_dir / "latest.json"
    if not latest.is_file():
        return None, None
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if payload.get("illustrative"):
        return None, None
    probe = payload.get("packing_probe") or {}
    if probe.get("manifest_sha256") != manifest_sha:
        return None, None
    p50 = probe.get("p50_ms")
    if isinstance(p50, bool) or not isinstance(p50, (int, float)):
        return None, None
    return float(p50), payload.get("date")


def packing_record(pooled, manifest_sha, previous, previous_date):
    delta = None
    if previous is not None and pooled["p50_ms"] is not None:
        delta = pooled["p50_ms"] - previous
    return {
        "model": "open-jev-27b-v1.1",
        "role": "packing_probe",
        "headline": False,
        "leaderboard_eligible": False,
        "workload": "packing_probe",
        "concurrency": 1,
        "manifest_sha256": manifest_sha,
        "p50_ms": pooled["p50_ms"],
        "p95_ms": pooled["p95_ms"],
        "sample_count": pooled["sample_count"],
        "evidence_complete": pooled["evidence_complete"],
        "failed_requests": pooled["failed_requests"],
        "delta_p50_ms_vs_previous": delta,
        "previous_date": previous_date,
        "raw_dirs": [item["output"] for item in pooled["rounds"]],
        "note": PACKING_NOTE,
    }


def assemble_report(
    *,
    started,
    finished,
    host,
    operator,
    sha,
    ref,
    dirty,
    gpu_info,
    workload,
    manifest_sha,
    manifest_requests,
    arms,
    packing,
    failures,
    issue,
    illustrative=False,
    note="",
):
    headline_p50 = {}
    headline_p95 = {}
    for arm in arms:
        if arm.get("headline") and arm.get("p50_ms") is not None:
            headline_p50[arm["id"]] = arm["p50_ms"]
            headline_p95[arm["id"]] = arm["p95_ms"]
    precisions = sorted(
        {arm.get("precision") for arm in arms if arm.get("precision")} or {"bf16"}
    )
    report = {
        "schema": SCHEMA,
        "illustrative": illustrative,
        "note": note,
        "date": started.astimezone(SHANGHAI).date().isoformat(),
        "timezone": "Asia/Shanghai",
        "started_at": started.astimezone(SHANGHAI).isoformat(),
        "finished_at": finished.astimezone(SHANGHAI).isoformat(),
        "host": host,
        "operator": operator,
        "git_sha": sha,
        "git_ref": ref,
        "git_dirty": dirty,
        "gpu": gpu_info.get("gpu", "unavailable"),
        "gpu_ids": gpu_info.get("gpu_ids", "unavailable"),
        "driver": gpu_info.get("driver", "unavailable"),
        "cuda": gpu_info.get("cuda", "unavailable"),
        "compute_cap": gpu_info.get("compute_cap", "unavailable"),
        "precision": ",".join(precisions),
        "latency_basis": LATENCY_BASIS,
        "quantile": QUANTILE,
        "comparable_to_h800_historical_board": False,
        "clock_note": CLOCK_NOTE,
        "workload": workload,
        "manifest_sha256": manifest_sha,
        "manifest_requests": manifest_requests,
        "headline_p50_ms": headline_p50,
        "headline_p95_ms": headline_p95,
        "arms": arms,
        "packing_probe": packing,
        "ok": not failures
        and all(
            arm.get("evidence_complete")
            for arm in arms
            if arm.get("role") == "headline"
        ),
        "failures": failures,
        "issue": issue,
    }
    missing = [key for key in REPORT_KEYS if key not in report]
    if missing:
        raise NightlyError(
            "internal summary is missing keys: " + ", ".join(missing),
            "内部 summary 缺少字段：" + "、".join(missing),
            code=1,
        )
    return report


def markdown_report(report):
    lines = [
        f"# hk2 nightly {report['date']}",
        "",
        "Headline / 每日主指标：Open-Jev-9B 与 Open-Jev-27B-v1.1，Public231，并发 C1。",
        "C8/C16 与 packing probe 不是 headline。Intern-Decision 仅在打开开关时出现。",
        report["clock_note"],
        "",
        f"- git `{report['git_sha']}` ref `{report['git_ref']}` dirty `{report['git_dirty']}`",
        f"- host `{report['host']}` gpu `{report['gpu']}` ids `{report['gpu_ids']}` driver `{report['driver']}` cuda `{report['cuda']}`",
        f"- workload `{report['workload']}` manifest `{report['manifest_sha256']}` requests `{report['manifest_requests']}`",
        f"- ok `{report['ok']}`",
        "",
        "| role | model | workload | C | p50 ms | p95 ms | ok/planned | eligible | note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for arm in report["arms"]:
        lines.append(
            "| {role} | {display} | {workload} | {concurrency} | {p50} | {p95} | {ok}/{planned} | {eligible} | {note} |".format(
                role=arm["role"],
                display=arm["display"],
                workload=arm["workload"],
                concurrency=arm["concurrency"],
                p50=arm["p50_ms"],
                p95=arm["p95_ms"],
                ok=arm["successful_requests"],
                planned=arm["planned_requests"],
                eligible=arm["leaderboard_eligible"],
                note=arm["note"],
            )
        )
    probe = report.get("packing_probe")
    if probe:
        lines.extend(
            [
                "",
                (
                    f"Packing probe p50 `{probe.get('p50_ms')}` p95 `{probe.get('p95_ms')}` "
                    f"delta_p50_ms_vs_previous `{probe.get('delta_p50_ms_vs_previous')}` "
                    f"(negative is faster). {probe.get('note')}"
                ),
            ]
        )
    raw_lines = [
        f"- {arm['id']} C{arm['concurrency']}: `{raw}`"
        for arm in report["arms"]
        for raw in arm.get("raw_dirs") or []
    ]
    if raw_lines:
        lines.extend(["", "Raw bench.py archives stay outside git:", *raw_lines])
    if report["failures"]:
        lines.extend(
            [
                "",
                "Failures:",
                *[f"- {item['arm']}: {item['message']}" for item in report["failures"]],
            ]
        )
    lines.extend(
        [
            "",
            "```json",
            json.dumps(issue_digest(report), indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def issue_digest(report):
    return {
        "schema": report["schema"],
        "date": report["date"],
        "git_sha": report["git_sha"],
        "git_ref": report["git_ref"],
        "host": report["host"],
        "gpu": report["gpu"],
        "gpu_ids": report["gpu_ids"],
        "workload": report["workload"],
        "manifest_sha256": report["manifest_sha256"],
        "latency_basis": report["latency_basis"],
        "comparable_to_h800_historical_board": False,
        "headline_p50_ms": report["headline_p50_ms"],
        "headline_p95_ms": report["headline_p95_ms"],
        "packing_probe": report["packing_probe"],
        "ok": report["ok"],
        "failures": report["failures"],
        "regression_only": [
            {
                "id": arm["id"],
                "concurrency": arm["concurrency"],
                "p50_ms": arm["p50_ms"],
                "p95_ms": arm["p95_ms"],
                "note": REGRESSION_NOTE,
            }
            for arm in report["arms"]
            if arm["role"] == "regression-only"
        ],
    }


def iter_result_files(results_dir):
    files = []
    for path in results_dir.glob("*.json"):
        if path.name == "latest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema") != SCHEMA or payload.get("illustrative"):
            continue
        files.append((payload.get("started_at") or "", path, payload))
    files.sort(key=lambda item: item[0])
    return files


def recent_markdown(results_dir, limit=30):
    rows = iter_result_files(results_dir)[-limit:]
    lines = [
        "# Recent hk2 nightly runs",
        "",
        "Day-over-day headline movement is `headline_p50_ms` for `open-jev-9b` and `open-jev-27b-v1.1`.",
        "Subtract the previous date. A negative difference is faster.",
        "Ignore `regression-only` and `packing_probe` when writing the latency leaderboard.",
        CLOCK_NOTE,
        "",
        "| date | sha | gpu | model | role | C | p50 ms | p95 ms | ok/planned |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _started, _path, payload in rows:
        sha = str(payload.get("git_sha", ""))[:12]
        for arm in payload.get("arms") or []:
            lines.append(
                f"| {payload.get('date')} | {sha} | {payload.get('gpu')} | {arm.get('display')} | {arm.get('role')} | {arm.get('concurrency')} | {arm.get('p50_ms')} | {arm.get('p95_ms')} | {arm.get('successful_requests')}/{arm.get('planned_requests')} |"
            )
        probe = payload.get("packing_probe")
        if probe:
            lines.append(
                f"| {payload.get('date')} | {sha} | {payload.get('gpu')} | Open-Jev-27B-v1.1 packing probe | packing_probe | 1 | {probe.get('p50_ms')} | {probe.get('p95_ms')} | delta {probe.get('delta_p50_ms_vs_previous')} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_results(results_dir, report):
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["date"]
    dated = results_dir / f"{stamp}.json"
    if dated.exists():
        clock = datetime.fromisoformat(report["started_at"]).strftime("%H%M%S")
        dated = results_dir / f"{stamp}T{clock}.json"
        suffix = 2
        while dated.exists():
            dated = results_dir / f"{stamp}T{clock}-{suffix}.json"
            suffix += 1
    write_json(dated, report)
    write_json(results_dir / "latest.json", report)
    (results_dir / "recent.md").write_text(
        recent_markdown(results_dir), encoding="utf-8"
    )
    (results_dir / f"{dated.stem}.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    return dated


def split_issue(value):
    text = value.strip()
    parts = urlsplit(text)
    if parts.scheme and parts.netloc:
        match = re.fullmatch(r"/([^/]+)/([^/]+)/issues/([1-9][0-9]*)", parts.path)
        if match is None or "github.com" not in parts.netloc:
            raise NightlyError(
                "OMNI_NIGHTLY_ISSUE URL is not a GitHub issue",
                "OMNI_NIGHTLY_ISSUE URL 不是 GitHub issue",
            )
        return match.group(3), f"{match.group(1)}/{match.group(2)}"
    if re.fullmatch(r"[1-9][0-9]*", text):
        return text, None
    raise NightlyError(
        "OMNI_NIGHTLY_ISSUE must be a positive issue number or a GitHub issue URL. Create the tracking issue first; there is no default.",
        "OMNI_NIGHTLY_ISSUE 必须是正整数的 issue 号或 GitHub issue URL。请先建追踪 issue；没有默认号。",
    )


def origin_slug(repo):
    url = git(repo, "remote", "get-url", "origin", timeout=30)
    match = re.search(r"github\.com[:/]([^/]+)/([^/\s.]+)", url)
    if match is None:
        raise NightlyError(
            "cannot parse the GitHub slug from origin",
            "无法从 origin 解析 GitHub 仓库名",
        )
    return f"{match.group(1)}/{match.group(2)}"


def post_issue(repo, issue, body_path, runner=None):
    number, slug = split_issue(issue)
    if slug is None:
        slug = origin_slug(repo)
    command = [
        "gh",
        "issue",
        "comment",
        number,
        "--repo",
        slug,
        "--body-file",
        str(body_path),
    ]
    run = runner or subprocess.run
    try:
        result = run(
            command,
            cwd=repo,
            timeout=60,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NightlyError(
            "gh issue comment timed out after 60s",
            "gh issue comment 超时（60 秒），已停止。本地 summary 已写入。",
            code=1,
        ) from exc
    except OSError as exc:
        raise NightlyError(
            f"could not run gh to comment on the tracking issue: {exc}. Local summaries were written.",
            f"无法运行 gh 评论追踪 issue：{exc}。本地 summary 已写入。",
            code=1,
        ) from exc
    if result.returncode != 0:
        raise NightlyError(
            "gh issue comment failed. Local summaries were written.",
            "gh issue comment 失败。本地 summary 已写入。",
            redact((result.stderr or result.stdout or "").strip()),
            code=1,
        )
    return number, slug


def build_release(repo, arch):
    if not arch:
        raise NightlyError(
            "OMNI_NIGHTLY_BUILD=1 requires OMNI_NIGHTLY_CUDA_ARCH (the device compute capability, for example 90). The harness does not guess. build.sh's own default is 89.",
            "OMNI_NIGHTLY_BUILD=1 时必须设置 OMNI_NIGHTLY_CUDA_ARCH（设备 compute capability，例如 90）。本程序不会猜测。build.sh 自己的默认值是 89。",
        )
    if not re.fullmatch(r"[0-9]{2,3}", arch):
        raise NightlyError(
            f"refusing CUDA arch {arch!r}",
            f"拒绝 CUDA arch {arch!r}",
        )
    commands = [
        ["src/backends/cuda/qwen3_5/build.sh", "target/release", arch],
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "-p",
            "omni-open-jev-native",
            "-p",
            "omni-jev",
        ],
    ]
    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=repo,
                timeout=3600,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NightlyError(
                f"{command[0]} timed out after 3600s",
                f"{command[0]} 超时（3600 秒），已停止。",
                code=1,
            ) from exc
        except OSError as exc:
            raise NightlyError(
                f"could not start {command[0]}: {exc}",
                f"无法启动 {command[0]}：{exc}",
                code=1,
            ) from exc
        if result.returncode != 0:
            raise NightlyError(
                f"{command[0]} failed with status {result.returncode}",
                f"{command[0]} 失败，退出码 {result.returncode}",
                code=1,
            )


def role_for(model, workload, concurrency):
    if concurrency != 1:
        return "regression-only"
    if workload != "public231":
        return "smoke" if workload == "smoke" else "custom"
    if model["cadence"] == "optional":
        return "optional"
    return "headline"


def note_for(role):
    return {
        "headline": "Concurrency-1 Public231 headline arm.",
        "optional": OPTIONAL_NOTE,
        "regression-only": REGRESSION_NOTE,
        "smoke": SMOKE_NOTE,
        "custom": CUSTOM_NOTE,
    }[role]


def prepare(args, repo):
    config = load_config(args.config or os.environ.get("OMNI_NIGHTLY_CONFIG"))
    include_intern = flag(
        args.intern, "OMNI_NIGHTLY_INTERN", bool(config.get("include_intern"))
    )
    models = selected_models(config, include_intern)
    require_weights(repo, models)
    for model in models:
        if model["launch"] == "attach-only":
            # Confirm the URL is present. The live GET happens when the arm starts.
            plan_service(model, repo, os.environ, probe=lambda _endpoint: {"ok": True})
    manifest, workload = resolve_manifest(args, config, repo)
    _cases, manifest_sha = load_manifest(manifest, workload)
    if flag(args.build, "OMNI_NIGHTLY_BUILD", False):
        build_release(repo, os.environ.get("OMNI_NIGHTLY_CUDA_ARCH", "").strip())
    require_binaries(repo, models)
    require_gpu_idle(models)
    regression = parse_concurrency_list(args.regression_concurrency)
    if regression is None:
        regression = parse_concurrency_list(
            os.environ.get("OMNI_NIGHTLY_REGRESSION_CONCURRENCY")
        )
    if regression is None:
        regression = [int(item) for item in config.get("regression_concurrency") or []]
        for item in regression:
            if item < 2:
                raise NightlyError(
                    "configured regression concurrency must be >= 2",
                    "配置里的回归并发必须 >= 2",
                )
    repeats = positive_int(
        args.repeats
        if args.repeats is not None
        else os.environ.get("OMNI_NIGHTLY_REPEATS", "1"),
        "repeats",
        3,
    )
    warmup = positive_int(
        args.warmup
        if args.warmup is not None
        else os.environ.get("OMNI_NIGHTLY_WARMUP", "5"),
        "warmup",
        20,
    )
    timeout = positive_int(
        args.timeout
        if args.timeout is not None
        else os.environ.get("OMNI_NIGHTLY_TIMEOUT", "120"),
        "timeout",
        600,
    )
    ready = positive_int(
        os.environ.get("OMNI_NIGHTLY_READY_TIMEOUT", "900"), "ready timeout", 3600
    )
    packing = (
        flag(args.packing, "OMNI_NIGHTLY_PACKING", True) and workload == "public231"
    )
    results = args.results or os.environ.get("OMNI_NIGHTLY_RESULTS")
    results_dir = (
        Path(os.path.expanduser(results))
        if results
        else repo / "benchmarks" / "nightly" / "results"
    )
    if not results_dir.is_absolute():
        results_dir = repo / results_dir
    raw = os.environ.get("OMNI_NIGHTLY_RAW", "~/experiments/system1-omni-nightly")
    raw_dir = Path(os.path.expanduser(raw))
    issue = (
        args.issue
        if args.issue is not None
        else os.environ.get("OMNI_NIGHTLY_ISSUE", "")
    )
    issue = issue.strip()
    if issue:
        split_issue(issue)
    ref = validate_ref(
        args.ref or os.environ.get("OMNI_NIGHTLY_REF") or config.get("ref") or "main"
    )
    return {
        "config": config,
        "models": models,
        "include_intern": include_intern,
        "manifest": manifest,
        "workload": workload,
        "manifest_sha": manifest_sha,
        "manifest_requests": len(_cases),
        "regression": regression,
        "repeats": repeats,
        "warmup": warmup,
        "timeout": timeout,
        "ready_timeout": ready,
        "packing": packing,
        "results_dir": results_dir,
        "raw_dir": raw_dir,
        "issue": issue,
        "ref": ref,
    }


def dry_plan(prepared, repo, sha, dirty):
    rows = []
    for model in prepared["models"]:
        configured = os.environ.get(model["endpoint_env"], "").strip()
        if configured:
            mode = "attach"
            endpoint = check_endpoint(configured)
            launch = None
        elif model["launch"] == "open-jev-native":
            mode = "launch"
            launch = public_launch(launch_spec(model, repo, visible_devices()))
            endpoint = launch["endpoint"]
        else:
            mode = "attach"
            endpoint = None
            launch = None
        rows.append(
            {
                "id": model["id"],
                "display": model["display"],
                "cadence": model["cadence"],
                "mode": mode,
                "endpoint": endpoint,
                "launch": launch,
            }
        )
    return {
        "dry_run": True,
        "schema": SCHEMA,
        "git_sha": sha,
        "git_dirty": dirty,
        "ref": prepared["ref"],
        "workload": prepared["workload"],
        "manifest": str(prepared["manifest"]),
        "manifest_sha256": prepared["manifest_sha"],
        "manifest_requests": prepared["manifest_requests"],
        "headline": prepared["workload"] == "public231",
        "intern": prepared["include_intern"],
        "models": rows,
        "regression_concurrency": prepared["regression"],
        "packing_probe": prepared["packing"],
        "issue": prepared["issue"] or None,
        "results_dir": str(prepared["results_dir"]),
        "raw_dir": str(prepared["raw_dir"]),
        "clock_note": CLOCK_NOTE,
    }


def execute(prepared, repo, sha, dirty):
    started = datetime.now(SHANGHAI)
    host = os.environ.get("OMNI_NIGHTLY_HOST") or socket.gethostname()
    operator = getpass.getuser()
    gpu_info = query_gpu(visible_devices())
    run_id = started.strftime("%Y%m%dT%H%M%S")
    raw_root = prepared["raw_dir"] / run_id
    raw_root.mkdir(parents=True, exist_ok=True)
    provenance = f"{sha} ref {prepared['ref']} dirty {dirty}"
    failures = []
    arms = []
    packing = None
    packing_sha = digest(HERE / "packing_probe.jsonl")
    previous, previous_date = previous_packing(prepared["results_dir"], packing_sha)
    smoke_manifest = repo / "benchmarks" / "smoke.jsonl"
    for model in prepared["models"]:
        server = None
        try:
            endpoint, attached, server = start_or_attach(
                model,
                repo,
                raw_root / "servers",
                prepared["ready_timeout"],
            )
            meta = metadata_for(model, repo, provenance, gpu_info, host)
            meta_path = raw_root / model["id"] / "metadata.json"
            write_json(meta_path, meta)
            if prepared["workload"] != "smoke":
                readiness = run_bench(
                    repo,
                    smoke_manifest,
                    endpoint,
                    model,
                    "feasibility",
                    1,
                    1,
                    prepared["timeout"],
                    meta_path,
                    raw_root / model["id"] / "readiness-smoke",
                    model.get("score_decimals", 4),
                )
                if not readiness["ok"]:
                    failures.append(
                        {
                            "arm": model["id"],
                            "message": f"smoke readiness failed; raw {readiness['output']}",
                        }
                    )
                    continue
            measured = []
            for index in range(prepared["repeats"]):
                measured.append(
                    run_bench(
                        repo,
                        prepared["manifest"],
                        endpoint,
                        model,
                        "measured",
                        1,
                        prepared["warmup"],
                        prepared["timeout"],
                        meta_path,
                        raw_root / model["id"] / f"c1-r{index + 1}",
                        model.get("score_decimals", 4),
                    )
                )
            role = role_for(model, prepared["workload"], 1)
            pooled = pool_runs(measured)
            record = arm_record(
                model,
                pooled,
                prepared["workload"],
                1,
                role,
                endpoint,
                attached,
                prepared["manifest_sha"],
                note_for(role),
            )
            record["precision"] = model.get("precision") or "bf16"
            record["raw_dirs"] = [item["output"] for item in pooled["rounds"]]
            arms.append(record)
            measured_ok = (
                pooled["evidence_complete"]
                and pooled["failed_requests"] == 0
                and pooled["sample_count"] > 0
            )
            if not measured_ok:
                failures.append(
                    {
                        "arm": model["id"],
                        "message": f"{role} C1 did not complete with zero failures",
                    }
                )
                continue
            for level in prepared["regression"]:
                regression_run = run_bench(
                    repo,
                    prepared["manifest"],
                    endpoint,
                    model,
                    "measured",
                    level,
                    prepared["warmup"],
                    prepared["timeout"],
                    meta_path,
                    raw_root / model["id"] / f"c{level}",
                    model.get("score_decimals", 4),
                )
                pooled_regression = pool_runs([regression_run])
                regression_arm = arm_record(
                    model,
                    pooled_regression,
                    prepared["workload"],
                    level,
                    "regression-only",
                    endpoint,
                    attached,
                    prepared["manifest_sha"],
                    REGRESSION_NOTE,
                )
                regression_arm["precision"] = model.get("precision") or "bf16"
                regression_arm["raw_dirs"] = [regression_run["output"]]
                arms.append(regression_arm)
                if not regression_run["ok"]:
                    failures.append(
                        {
                            "arm": f"{model['id']}-c{level}",
                            "message": "regression gate failed",
                        }
                    )
            if (
                prepared["packing"]
                and model.get("packing_probe")
                and record.get("evidence_complete")
            ):
                probe_run = run_bench(
                    repo,
                    HERE / "packing_probe.jsonl",
                    endpoint,
                    model,
                    "measured",
                    1,
                    1,
                    prepared["timeout"],
                    meta_path,
                    raw_root / model["id"] / "packing-probe",
                    model.get("score_decimals", 4),
                )
                packing = packing_record(
                    pool_runs([probe_run]), packing_sha, previous, previous_date
                )
                if not probe_run["ok"]:
                    failures.append(
                        {"arm": model["id"], "message": "packing probe failed"}
                    )
        except NightlyError as exc:
            failures.append({"arm": model["id"], "message": str(exc)})
        finally:
            if server is not None:
                server.close()
    finished = datetime.now(SHANGHAI)
    report = assemble_report(
        started=started,
        finished=finished,
        host=host,
        operator=operator,
        sha=sha,
        ref=prepared["ref"],
        dirty=dirty,
        gpu_info=gpu_info,
        workload=prepared["workload"],
        manifest_sha=prepared["manifest_sha"],
        manifest_requests=prepared["manifest_requests"],
        arms=arms,
        packing=packing,
        failures=failures,
        issue=prepared["issue"] or None,
        note="Local hk2 nightly. Headline numbers are Public231 C1 only.",
    )
    dated = write_results(prepared["results_dir"], report)
    mirror = prepared["raw_dir"] / "summaries"
    write_json(mirror / dated.name, report)
    write_json(mirror / "latest.json", report)
    comment_path = raw_root / "issue-comment.md"
    comment_path.write_text(markdown_report(report), encoding="utf-8")
    if prepared["issue"]:
        try:
            post_issue(repo, prepared["issue"], comment_path)
        except NightlyError as exc:
            report["ok"] = False
            report["failures"].append({"arm": "issue-comment", "message": str(exc)})
            write_json(dated, report)
            write_json(prepared["results_dir"] / "latest.json", report)
            failures.append(report["failures"][-1])
    if failures:
        names = ", ".join(item["arm"] for item in failures)
        raise NightlyError(
            f"nightly finished with failures: {names}. Summaries were written to {dated}.",
            f"nightly 已结束，但有失败项：{names}。summary 已写入 {dated}。",
            code=1,
        )
    print(
        json.dumps(
            {
                "ok": True,
                "summary": str(dated),
                "latest": str(prepared["results_dir"] / "latest.json"),
            },
            indent=2,
        )
    )
    return report


def self_check():
    config = load_config(None)
    nightly = selected_models(config, include_intern=False)
    optional = selected_models(config, include_intern=True)
    nightly_ids = [model["id"] for model in nightly]
    optional_ids = [model["id"] for model in optional]
    if nightly_ids != ["open-jev-9b", "open-jev-27b-v1.1"]:
        raise NightlyError(
            f"unexpected headline models: {nightly_ids}",
            f"headline 模型与约定不符：{nightly_ids}",
        )
    if optional_ids != [
        "open-jev-9b",
        "open-jev-27b-v1.1",
        "intern-decision-0.8b",
        "intern-decision-2b",
        "intern-decision-4b",
    ]:
        raise NightlyError(
            f"unexpected optional set: {optional_ids}",
            f"可选模型与约定不符：{optional_ids}",
        )
    shell = (HERE / "run_nightly.sh").read_text(encoding="utf-8")
    positions = [shell.index(item) for item in CONVENTIONAL_REPOS]
    if positions != sorted(positions):
        raise NightlyError(
            "shell repo search order drifted from CONVENTIONAL_REPOS",
            "shell 仓库搜索顺序与 CONVENTIONAL_REPOS 不一致",
        )
    bench = load_bench()
    cases = bench.load_cases(HERE / "packing_probe.jsonl")
    if len(cases) != 4:
        raise NightlyError(
            "packing probe must stay a four-request smoke",
            "packing probe 必须保持为 4 条请求",
        )
    example = json.loads((HERE / "example-latest.json").read_text(encoding="utf-8"))
    if example.get("illustrative") is not True or example.get("schema") != SCHEMA:
        raise NightlyError(
            "example-latest.json must be an illustrative schema sample",
            "example-latest.json 必须是带 illustrative 标记的 schema 样例",
        )
    missing = [key for key in REPORT_KEYS if key not in example]
    if missing:
        raise NightlyError(
            "example-latest.json is missing keys: " + ", ".join(missing),
            "example-latest.json 缺少字段：" + "、".join(missing),
        )
    print("self-check ok")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workload", choices=("public231", "smoke", "custom"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--ref")
    parser.add_argument("--issue", default=None)
    parser.add_argument("--regression-concurrency", dest="regression_concurrency")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--update", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--intern", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--packing", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--build", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
    return parser.parse_args(argv)


def acquire_lock(path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise NightlyError(
            "another nightly is already running; this one exits.",
            "已有 nightly 在运行，本次退出，不会等待。",
        ) from exc
    return handle


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    lock = None
    try:
        if args.self_check:
            self_check()
            return 0
        repo = (args.repo or REPO_ROOT).resolve()
        if not (repo / "benchmarks" / "bench.py").is_file() and not args.dry_run:
            raise NightlyError(
                f"{repo} does not look like a system1-omni checkout",
                f"{repo} 不像 system1-omni 仓库",
            )
        if not args.no_lock:
            lock_path = Path(
                os.path.expanduser(
                    os.environ.get(
                        "OMNI_NIGHTLY_LOCK",
                        "~/experiments/system1-omni-nightly/nightly.lock",
                    )
                )
            )
            lock = acquire_lock(lock_path)
        update = flag(args.update, "OMNI_NIGHTLY_UPDATE", True)
        ref = validate_ref(args.ref or os.environ.get("OMNI_NIGHTLY_REF") or "main")
        if update:
            sha, dirty = maybe_update(repo, ref, True)
        else:
            sha, dirty = read_provenance(repo, required=False)
        prepared = prepare(args, repo)
        if args.dry_run:
            print(
                json.dumps(
                    dry_plan(prepared, repo, sha, dirty), indent=2, ensure_ascii=False
                )
            )
            return 0
        if sha == "unknown":
            raise NightlyError(
                "git SHA is unknown; the nightly dashboard requires a checkout revision",
                "无法得到 git SHA；nightly dashboard 需要仓库版本。",
            )
        execute(prepared, repo, sha, dirty)
        return 0
    except NightlyError as exc:
        sys.stderr.write(str(exc))
        return exc.code
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
