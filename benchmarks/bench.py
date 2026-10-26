#!/usr/bin/env python3
"""Replay frozen System One requests against an already running GPU server."""

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import httpx


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def response_digest(text):
    if text is None:
        return None
    if not isinstance(text, str):
        raise ValueError("raw response must be text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


RESPONSE_FORMAT = "httpx-content-base64-utf8-v1"
LEGACY_RESPONSE_FORMAT = "httpx-text-v1"

# Public231 fast-screen. One measured round samples these prompt-token buckets
# (inclusive lower bound, exclusive upper bound; 2048+ is unbounded). Selection
# rank is SHA-256 of "{seed}\\n{bucket}\\n{id}"; the quota of lowest ranks is
# replayed in manifest order. Seed 231 is the fixed draw.
FAST_SCREEN_SEED = 231
FAST_SCREEN_RELATIVE = 0.02
FAST_SCREEN_ABSOLUTE_MS = 0.5
FAST_SCREEN_DROP_NAME = "fast_screen_drop.json"
# Placeholder until the primary SHA is screened on these same request IDs.
SCREEN_PRIMARY_TBD = "screen primary TBD — measure on primary SHA same request IDs"
# Full-C1 pooled pairs. These are not screen baselines. StartLux Instant's
# pooled primary is the first entry; do not copy it onto a fast-screen.
FULL_C1_POOLED_NOT_SCREEN = ((7.338, 51.816),)
# Quotas sum to 58 when every bucket is full. 1024-2047 is 6 (InferRSI
# 2026-10-26). A shorter bucket is capped, not filled from another bucket.
FAST_SCREEN_BUCKETS = (
    ("0-127", 0, 128, 10),
    ("128-255", 128, 256, 10),
    ("256-511", 256, 512, 8),
    ("512-1023", 512, 1024, 8),
    ("1024-2047", 1024, 2048, 6),
    ("2048+", 2048, None, 16),
)
FAST_SCREEN_TAIL = "2048+"


def saved_response_format(config):
    value = config.get("response_format", LEGACY_RESPONSE_FORMAT)
    if value not in (RESPONSE_FORMAT, LEGACY_RESPONSE_FORMAT):
        raise ValueError("unsupported saved response format")
    return value


def response_body(record, response_format):
    if response_format == LEGACY_RESPONSE_FORMAT:
        body = record["response"]
        checksum = response_digest(body)
    else:
        encoded = record["response_bytes_base64"]
        if encoded is not None and not isinstance(encoded, str):
            raise ValueError("saved response base64 must be text")
        body = base64.b64decode(encoded, validate=True) if encoded is not None else None
        checksum = hashlib.sha256(body).hexdigest() if body is not None else None
        text = body.decode("utf-8", errors="replace") if body is not None else None
        if record["response"] != text:
            raise ValueError("saved response text does not match its bytes")
        if body is None and record["status"] is not None:
            raise ValueError("HTTP response is missing its saved bytes")
    if record["response_sha256"] != checksum:
        raise ValueError("raw response checksum mismatch")
    return body


def response_payload(body):
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


def load_cases(path):
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("manifest must contain nonempty, unique request IDs")
    for case in cases:
        request = case["request"]
        if "model" in request:
            raise ValueError(
                "set the backend model alias with --model, not in the manifest"
            )
        if "state" not in request or not request.get("questions"):
            raise ValueError("each request needs state and questions")
        if set(case["expected"]) != set(request["questions"]):
            raise ValueError("expected labels must cover every question")
        for qid, question in request["questions"].items():
            kind, target = question["type"], case["expected"][qid]
            if kind == "choice":
                criteria = question["criteria"]
                if (
                    not isinstance(criteria, dict)
                    or not criteria
                    or target not in criteria
                ):
                    raise ValueError(
                        "Choice requires a criteria map and a matching label"
                    )
            elif kind == "noul":
                if not isinstance(target, bool):
                    raise ValueError("Noul target must be boolean")
            elif kind == "score":
                criteria = question["criteria"]
                if not isinstance(criteria, list) or len(criteria) < 2:
                    raise ValueError("Score requires at least two ordered criteria")
                if not number(target) or not 0 <= target <= len(criteria) - 1:
                    raise ValueError("Score target must be a finite level index")
            else:
                raise ValueError(f"unsupported question type: {kind}")
    return cases


def number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


WIRE_DECIMALS = 4  # workers round probabilities on the wire to four decimal places
VALIDATION_PROTOCOL = "probabilities-4dp-model-score-v1"
LEGACY_VALIDATION_PROTOCOL = "systemone-4dp-score-abs1e-4-v1"


def validation_settings(config):
    version = config.get("validation_protocol", LEGACY_VALIDATION_PROTOCOL)
    protocol = config.get("protocol", "systemone")
    decimals = config.get("score_decimals", 4)
    if version not in (LEGACY_VALIDATION_PROTOCOL, VALIDATION_PROTOCOL):
        raise ValueError("unsupported saved validation protocol")
    if protocol not in ("systemone", "jev-vl"):
        raise ValueError("unsupported wire protocol")
    if type(decimals) is not int or decimals not in (2, 3, 4):
        raise ValueError("score_decimals must be 2, 3 or 4")
    if version == LEGACY_VALIDATION_PROTOCOL and (
        protocol != "systemone" or decimals != 4
    ):
        raise ValueError("legacy validation requires System One and four-decimal Score")
    return protocol, decimals, version


def sum_allowance(count):
    """How far `count` wire probabilities may miss a normalized sum: each rounded value
    carries up to half a decimal step, so the allowance grows with the label count. A
    flat 1e-4 rejected four-value distributions summing to 0.9999 at the float boundary
    (PR #40, first GPU run)."""
    return count * 0.5 * 10**-WIRE_DECIMALS + 1e-9


def score_allowance(count, score_decimals=4):
    """Bound independent rounding of the scalar and its weighted probabilities."""
    return (
        0.5 * 10**-score_decimals
        + 0.5 * 10**-WIRE_DECIMALS * count * (count - 1) / 2
        + 1e-9
    )


def zero_score_fit(answer, probabilities):
    """Recognize Decider's isolated Score response when no level has any fit."""
    level_fit = answer.get("level_fit")
    return (
        answer.get("type") == "score"
        and all(p == 0 for p in probabilities.values())
        and isinstance(level_fit, dict)
        and set(level_fit) == set(probabilities)
        and all(v == 0 and number(v) for v in level_fit.values())
        and all(
            answer.get(key) == 0 and number(answer[key])
            for key in ("score", "confidence", "x_p_max", "fit_mass")
        )
        and answer.get("certainty") == 1
        and number(answer["certainty"])
    )


def read_answers(
    case, payload, score_decimals=4, validation_protocol=VALIDATION_PROTOCOL
):
    """Validate wire results and normalize probabilities by label, never by order."""
    answers = payload["answers"]
    if not isinstance(answers, dict) or set(answers) != set(
        case["request"]["questions"]
    ):
        raise ValueError("response question IDs do not match the request")
    normalized = {}
    for qid, question in case["request"]["questions"].items():
        answer = answers[qid]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError(
                "response answer must be an object with the requested type"
            )
        if kind == "noul":
            value = answer["noul"]
            if not number(value):
                raise ValueError("Noul probability must be finite")
            probabilities = {"false": 1 - value, "true": value}
        else:
            labels = (
                list(question["criteria"])
                if kind == "choice"
                else [str(i) for i in range(len(question["criteria"]))]
            )
            probabilities = answer["probabilities"]
            if set(probabilities) != set(labels):
                raise ValueError("response probability labels do not match criteria")
            value = answer[kind]
        if any(not number(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("probabilities must be finite and in [0, 1]")
        total = math.fsum(probabilities.values())
        if abs(total - 1) > sum_allowance(len(probabilities)) and not (
            validation_protocol == VALIDATION_PROTOCOL
            and kind == "score"
            and total == 0
            and zero_score_fit(answer, probabilities)
        ):
            raise ValueError("probabilities must sum to one")
        if kind == "choice":
            if value not in probabilities or probabilities[value] != max(
                probabilities.values()
            ):
                raise ValueError("choice must have maximal probability")
        elif not number(value):
            raise ValueError("decision value must be finite")
        if kind == "score":
            if validation_protocol == LEGACY_VALIDATION_PROTOCOL:
                expected = sum(int(k) * p for k, p in probabilities.items())
                valid = math.isclose(value, expected, abs_tol=1e-4)
            else:
                if not 0 <= value <= len(probabilities) - 1:
                    raise ValueError("score must be in the level range")
                expected = math.fsum(int(k) * p for k, p in probabilities.items())
                valid = math.isclose(
                    value,
                    expected,
                    rel_tol=0,
                    abs_tol=score_allowance(len(probabilities), score_decimals),
                )
            if not valid:
                raise ValueError("score must equal the expected level")
        normalized[qid] = {"value": value, "probabilities": probabilities}
    return normalized


def nearest_rank_p95(samples):
    if not samples:
        return None
    ordered = sorted(samples)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def prompt_token_count(case):
    if "prompt_tokens" not in case:
        raise ValueError(
            f"case {case['id']!r} needs integer prompt_tokens for fast-screen"
        )
    value = case["prompt_tokens"]
    if type(value) is not int or value < 0:
        raise ValueError(
            f"case {case['id']!r} prompt_tokens must be a nonnegative integer"
        )
    return value


def fast_screen_bucket(tokens):
    for name, lo, hi, _quota in FAST_SCREEN_BUCKETS:
        if tokens >= lo and (hi is None or tokens < hi):
            return name
    raise ValueError(f"prompt_tokens {tokens} is outside fast-screen buckets")


def _fast_screen_rank(seed, bucket, case_id):
    return hashlib.sha256(f"{seed}\n{bucket}\n{case_id}".encode()).hexdigest()


def fast_screen_quotas():
    return {name: quota for name, _lo, _hi, quota in FAST_SCREEN_BUCKETS}


def fast_screen_sample(cases, seed=FAST_SCREEN_SEED):
    """Cap each bucket to its quota. A short bucket keeps every case.

    Nothing is borrowed from another bucket. `quota_shortfall` lists each
    bucket whose pool was smaller than its quota. 2048+ below 16 does not
    raise here; the saved decision is inconclusive and cannot early-DROP.
    """
    grouped = {name: [] for name, *_rest in FAST_SCREEN_BUCKETS}
    for case in cases:
        grouped[fast_screen_bucket(prompt_token_count(case))].append(case)
    chosen = set()
    counts = {}
    shortfall = []
    for name, _lo, _hi, quota in FAST_SCREEN_BUCKETS:
        pool = grouped[name]
        ranked = sorted(
            pool,
            key=lambda case: (_fast_screen_rank(seed, name, case["id"]), case["id"]),
        )
        picked = ranked[:quota]
        counts[name] = len(picked)
        if len(pool) < quota:
            shortfall.append({"name": name, "quota": quota, "found": len(pool)})
        chosen.update(case["id"] for case in picked)
    if not chosen:
        raise ValueError("fast-screen selected no requests")
    return [case for case in cases if case["id"] in chosen], counts, shortfall


def latency_clearly_worse(baseline_ms, tip_ms):
    """True when tip - baseline is at least +2% or +0.5 ms."""
    if (
        type(baseline_ms) is bool
        or type(tip_ms) is bool
        or not isinstance(baseline_ms, (int, float))
        or not isinstance(tip_ms, (int, float))
        or not math.isfinite(baseline_ms)
        or not math.isfinite(tip_ms)
        or baseline_ms <= 0
    ):
        raise ValueError(
            "fast-screen latencies must be finite and the baseline must be positive"
        )
    # Compare on the tip side so an exact +2% or +0.5 ms boundary stays worse.
    # Subtracting first can land just below the threshold for the same decimal.
    return tip_ms >= baseline_ms * (
        1 + FAST_SCREEN_RELATIVE
    ) or tip_ms >= baseline_ms + FAST_SCREEN_ABSOLUTE_MS


def fast_screen_decision(baseline_p50, baseline_p95, tip_p50, tip_p95):
    """DROP only when both percentiles are clearly worse. Never returns keep."""
    p50_worse = latency_clearly_worse(baseline_p50, tip_p50)
    p95_worse = latency_clearly_worse(baseline_p95, tip_p95)
    if p50_worse and p95_worse:
        return "drop"
    return "full_c1"


def is_full_c1_pooled_baseline(p50_ms, p95_ms):
    """True for published full-C1 pooled pairs that must not be a screen baseline."""
    if type(p50_ms) is bool or type(p95_ms) is bool:
        return False
    if not isinstance(p50_ms, (int, float)) or not isinstance(p95_ms, (int, float)):
        return False
    return any(
        math.isclose(p50_ms, pooled_p50, rel_tol=0, abs_tol=5e-4)
        and math.isclose(p95_ms, pooled_p95, rel_tol=0, abs_tol=5e-4)
        for pooled_p50, pooled_p95 in FULL_C1_POOLED_NOT_SCREEN
    )


def refuse_full_c1_pooled_baseline(p50_ms, p95_ms):
    if is_full_c1_pooled_baseline(p50_ms, p95_ms):
        raise ValueError(
            "refusing full-C1 pooled numbers as a fast-screen baseline "
            f"({p50_ms} / {p95_ms} ms). {SCREEN_PRIMARY_TBD}"
        )


def latency_by_prompt_bucket(cases, records):
    """Per-bucket p50/p95 when every case carries prompt_tokens. Else None."""
    if any("prompt_tokens" not in case for case in cases):
        return None
    by_id = {case["id"]: prompt_token_count(case) for case in cases}
    grouped = {name: [] for name, *_rest in FAST_SCREEN_BUCKETS}
    for record in records:
        if record.get("latency_ms") is None:
            continue
        grouped[fast_screen_bucket(by_id[record["id"]])].append(record["latency_ms"])
    report = {}
    for name, _lo, _hi, _quota in FAST_SCREEN_BUCKETS:
        samples = grouped[name]
        report[name] = {
            "samples": len(samples),
            "p50_ms": statistics.median(samples) if samples else None,
            "p95_ms": nearest_rank_p95(samples),
        }
    return report


def _refuse_numeric_screen_baseline(args):
    """Typed percentiles are how a full-C1 pooled pair gets reused. Refuse them."""
    if (
        getattr(args, "baseline_p50_ms", None) is not None
        or getattr(args, "baseline_p95_ms", None) is not None
    ):
        raise ValueError(
            "refusing --baseline-p50-ms and --baseline-p95-ms. "
            "Full-C1 pooled percentiles, including StartLux Instant "
            "7.338/51.816, are not a screen baseline. " + SCREEN_PRIMARY_TBD
        )


def _require_fast_screen_shape(args):
    if args.phase != "measured":
        raise ValueError("fast-screen requires --phase measured")
    if args.concurrency != 1:
        raise ValueError("fast-screen requires --concurrency 1")
    if args.warmup != 5:
        raise ValueError("fast-screen requires --warmup 5")


def _valid_dropped_at(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) == value


def load_same_request_screen_baseline(path, selected_ids, parent_manifest_sha256, tip_gpu):
    """Baseline is the primary SHA's own screen of these exact request IDs."""
    if not isinstance(path, Path) or not path.is_dir():
        raise ValueError("baseline screen directory is missing")
    if not (path / "config.json").is_file():
        raise ValueError("baseline screen is missing config.json")
    config = json.loads((path / "config.json").read_text())
    screen = config.get("fast_screen")
    if not screen:
        raise ValueError(
            "a full-C1 directory is not a fast-screen baseline. " + SCREEN_PRIMARY_TBD
        )
    if screen.get("role") != "screen_primary":
        raise ValueError(
            "baseline screen must be the primary SHA's own fast-screen "
            "(role screen_primary) of these same request IDs, not a tip comparison "
            "and not full-C1 pooled percentiles. " + SCREEN_PRIMARY_TBD
        )
    if screen.get("selected_ids") != list(selected_ids):
        raise ValueError(
            "baseline screen request IDs do not match this fast-screen"
        )
    if screen.get("seed") != FAST_SCREEN_SEED:
        raise ValueError("baseline screen seed does not match")
    if screen.get("parent_manifest_sha256") != parent_manifest_sha256:
        raise ValueError("baseline screen manifest does not match")
    if screen.get("quotas") != fast_screen_quotas():
        raise ValueError("baseline screen quotas do not match")
    metadata = config.get("metadata") or {}
    gpu = metadata.get("gpu")
    if gpu != tip_gpu:
        raise ValueError("baseline screen gpu does not match this run")
    revision = metadata.get("runtime_revision")
    if not revision:
        raise ValueError("baseline screen is missing runtime_revision")
    summary = summarize_saved(path)
    if summary.get("fast_screen_decision") != "screen_primary":
        raise ValueError(
            "baseline screen is not a completed screen-primary measurement. "
            + SCREEN_PRIMARY_TBD
        )
    p50 = summary["successful_latency_p50_ms"]
    p95 = summary["successful_latency_p95_ms"]
    refuse_full_c1_pooled_baseline(p50, p95)
    return {
        "baseline_kind": "same_request_id_screen",
        "baseline_runtime_revision": revision,
        "baseline_gpu": gpu,
        "baseline_p50_ms": p50,
        "baseline_p95_ms": p95,
        "baseline_selected_ids": list(selected_ids),
    }


def manifest_subset_bytes(path, selected_ids):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    by_id = {}
    for line in lines:
        item = json.loads(line)
        by_id[item["id"]] = line
    missing = [case_id for case_id in selected_ids if case_id not in by_id]
    if missing:
        raise ValueError("fast-screen selection is missing manifest lines")
    return "".join(by_id[case_id] + "\n" for case_id in selected_ids).encode()


def _check_saved_fast_screen(screen, cases, config):
    if screen.get("seed") != FAST_SCREEN_SEED:
        raise ValueError("unsupported fast-screen seed")
    if (
        screen.get("relative_worse") != FAST_SCREEN_RELATIVE
        or screen.get("absolute_worse_ms") != FAST_SCREEN_ABSOLUTE_MS
    ):
        raise ValueError("fast-screen thresholds do not match this runner")
    _validate_screen_counts(screen, cases)
    if screen.get("selected_ids") != [case["id"] for case in cases]:
        raise ValueError("fast-screen selection does not match the saved manifest")
    role = screen.get("role")
    if role == "screen_primary":
        if "baseline_p50_ms" in screen or "baseline_p95_ms" in screen:
            raise ValueError(
                "a screen-primary measurement must not carry typed baseline percentiles. "
                + SCREEN_PRIMARY_TBD
            )
        return
    if role != "tip":
        raise ValueError("unsupported fast-screen role")
    if screen.get("baseline_kind") != "same_request_id_screen":
        raise ValueError(
            "tip fast-screen baseline must be the same request IDs. "
            + SCREEN_PRIMARY_TBD
        )
    if "baseline_p50_ms" not in screen or "baseline_p95_ms" not in screen:
        raise ValueError("saved tip fast-screen is missing its same-ID baseline")
    refuse_full_c1_pooled_baseline(screen["baseline_p50_ms"], screen["baseline_p95_ms"])
    if screen.get("baseline_selected_ids") != screen["selected_ids"]:
        raise ValueError("saved screen baseline is not the same request IDs")
    if screen.get("baseline_gpu") != config.get("metadata", {}).get("gpu"):
        raise ValueError("saved screen baseline gpu does not match this run")
    if not screen.get("baseline_runtime_revision"):
        raise ValueError("saved screen baseline is missing the primary SHA")


def _validate_screen_counts(screen, cases):
    """Taken counts stay inside each quota. A gap is a recorded shortfall."""
    quotas = fast_screen_quotas()
    if screen.get("quotas") != quotas:
        raise ValueError("fast-screen quotas do not match this runner")
    counts = screen.get("bucket_counts")
    if not isinstance(counts, dict) or set(counts) != set(quotas):
        raise ValueError("fast-screen bucket counts do not match the protocol")
    shortfall = screen.get("quota_shortfall")
    if not isinstance(shortfall, list):
        raise ValueError("fast-screen quota_shortfall is missing")
    short_by_name = {}
    for item in shortfall:
        if not isinstance(item, dict):
            raise ValueError("fast-screen quota_shortfall has a bad bucket")
        name = item.get("name")
        if name not in quotas or name in short_by_name:
            raise ValueError("fast-screen quota_shortfall has a bad bucket")
        if item.get("quota") != quotas[name]:
            raise ValueError("fast-screen quota_shortfall quota does not match")
        found = item.get("found")
        if type(found) is not int or found < 0 or found >= quotas[name]:
            raise ValueError("fast-screen quota_shortfall found is not below quota")
        short_by_name[name] = found
    order = [name for name, *_rest in FAST_SCREEN_BUCKETS if name in short_by_name]
    if [item["name"] for item in shortfall] != order:
        raise ValueError("fast-screen quota_shortfall is out of bucket order")
    for name, quota in quotas.items():
        taken = counts[name]
        if type(taken) is not int or taken < 0 or taken > quota:
            raise ValueError("fast-screen bucket count is outside the quota")
        if name in short_by_name:
            if taken != short_by_name[name]:
                raise ValueError(
                    "fast-screen shortfall does not match the taken count"
                )
        elif taken != quota:
            raise ValueError(
                "fast-screen bucket is below quota without quota_shortfall"
            )
    if sum(counts.values()) != len(cases):
        raise ValueError("fast-screen manifest does not match the taken bucket counts")
    actual = {name: 0 for name in quotas}
    for case in cases:
        actual[fast_screen_bucket(prompt_token_count(case))] += 1
    if actual != counts:
        raise ValueError("fast-screen cases do not match bucket_counts")


def _tail_short(screen):
    """2048+ below 16 makes the whole screen inconclusive. No early DROP."""
    counts = screen.get("bucket_counts") or {}
    taken = counts.get(FAST_SCREEN_TAIL)
    if type(taken) is int and taken < fast_screen_quotas()[FAST_SCREEN_TAIL]:
        return True
    for item in screen.get("quota_shortfall") or []:
        if isinstance(item, dict) and item.get("name") == FAST_SCREEN_TAIL:
            found = item.get("found")
            if type(found) is not int or found < fast_screen_quotas()[FAST_SCREEN_TAIL]:
                return True
    return False


def _saved_fast_screen_decision(screen, summary):
    complete = (
        summary["evidence_complete"]
        and summary["failed_requests"] == 0
        and summary["incomplete_requests"] == 0
        and summary["unattempted_requests"] == 0
        and summary["successful_requests"] == summary["planned_requests"]
        and summary["successful_latency_p50_ms"] is not None
        and summary["successful_latency_p95_ms"] is not None
    )
    if not complete:
        return "inconclusive"
    if _tail_short(screen):
        return "inconclusive"
    if screen.get("role") == "screen_primary":
        return "screen_primary"
    return fast_screen_decision(
        screen["baseline_p50_ms"],
        screen["baseline_p95_ms"],
        summary["successful_latency_p50_ms"],
        summary["successful_latency_p95_ms"],
    )


def _attach_dropped_at(path, screen, summary):
    drop_path = path / FAST_SCREEN_DROP_NAME
    if summary.get("fast_screen_decision") != "drop":
        if drop_path.exists():
            raise ValueError("fast_screen_drop.json is only written for a DROP")
        return
    if not drop_path.is_file():
        return
    record = json.loads(drop_path.read_text())
    if record.get("decision") != "drop" or not _valid_dropped_at(record.get("dropped_at")):
        raise ValueError("fast-screen DROP record is missing dropped_at")
    if record.get("selected_ids") != screen["selected_ids"]:
        raise ValueError("fast-screen DROP record does not match the screen request IDs")
    summary["dropped_at"] = record["dropped_at"]


def quality(cases, records):
    values = {"choice_accuracy": [], "noul_accuracy": [], "score_mae": []}
    by_id = {r["id"]: r for r in records}
    for case in cases:
        record = by_id.get(case["id"])
        if record is None or record["error"] is not None:
            continue
        for qid, target in case["expected"].items():
            value = record["answers"][qid]["value"]
            kind = case["request"]["questions"][qid]["type"]
            if kind == "score":
                values["score_mae"].append(abs(value - target))
            else:
                prediction = value >= 0.5 if kind == "noul" else value
                values[kind + "_accuracy"].append(float(prediction == target))
    return {
        name: {"value": statistics.mean(v) if v else None, "count": len(v)}
        for name, v in values.items()
    }


def summarize(cases, records, elapsed, attempted_ids=None):
    attempted = set(
        attempted_ids if attempted_ids is not None else (r["id"] for r in records)
    )
    completed = [r for r in records if r.get("state", "completed") == "completed"]
    successful = [r for r in completed if r["error"] is None]
    failed = [r for r in completed if r["error"] is not None]
    completed_ids = {r["id"] for r in completed}
    incomplete = attempted - completed_ids
    latencies = sorted(r["latency_ms"] for r in successful)
    failed_latencies = sorted(r["latency_ms"] for r in failed)
    decisions = sum(len(r["answers"]) for r in successful)
    counts = {
        "planned": 0,
        "attempted": 0,
        "unattempted": 0,
        "incomplete": 0,
        "failed": 0,
    }
    failed_ids = {r["id"] for r in failed}
    for case in cases:
        size = len(case["request"]["questions"])
        counts["planned"] += size
        counts["attempted" if case["id"] in attempted else "unattempted"] += size
        if case["id"] in incomplete:
            counts["incomplete"] += size
        if case["id"] in failed_ids:
            counts["failed"] += size
    errors = {}
    for record in failed:
        key = record["error_kind"]
        errors[key] = errors.get(key, 0) + 1
    result = {
        "requests": len(records),
        "planned_requests": len(cases),
        "attempted_requests": len(attempted),
        "unattempted_requests": len(cases) - len(attempted),
        "incomplete_requests": len(incomplete),
        "completed_requests": len(completed),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "errors": errors,
        **{name + "_decisions": count for name, count in counts.items()},
        "successful_decisions": decisions,
        "completed_decisions": decisions + counts["failed"],
        "wall_seconds": elapsed,
        "requests_per_second": len(successful) / elapsed if elapsed else None,
        "decisions_per_second": decisions / elapsed if elapsed else None,
        "evidence_complete": elapsed is not None and len(completed) == len(cases),
        "latency_basis": "post_to_body_json_and_schema_validation",
        "successful_latency_p50_ms": statistics.median(latencies)
        if latencies
        else None,
        "successful_latency_p95_ms": nearest_rank_p95(latencies),
        "failed_latency_p50_ms": statistics.median(failed_latencies)
        if failed_latencies
        else None,
        "failed_latency_p95_ms": nearest_rank_p95(failed_latencies),
        "quality_on_successful_requests": quality(cases, records),
    }
    buckets = latency_by_prompt_bucket(cases, successful)
    if buckets is not None:
        result["latency_by_prompt_bucket"] = buckets
    return result


def wire_request(case, model, protocol):
    if protocol == "systemone":
        return {**case["request"], "model": model}
    if protocol != "jev-vl":
        raise ValueError("unsupported wire protocol")
    questions = case["request"]["questions"]
    if len(questions) != 1:
        raise ValueError("JEV-VL requires one question per request")
    question = next(iter(questions.values()))
    if question["type"] != "choice":
        raise ValueError("the JEV-VL benchmark adapter supports choice only")
    state = case["request"]["state"]
    if isinstance(state, dict) and set(state) == {"image"}:
        # JEV-VL treats a plain object as JSON text; media must be a parts list.
        state = [state]
    return {
        "model": model,
        "kind": "choice",
        "state": state,
        "question": question["instructions"],
        "options": [f"{k}: {v}" for k, v in question["criteria"].items()],
    }


def wire_answers(
    case,
    payload,
    model,
    protocol,
    score_decimals=4,
    validation_protocol=VALIDATION_PROTOCOL,
):
    if protocol == "systemone":
        return read_answers(case, payload, score_decimals, validation_protocol)
    request = wire_request(case, model, protocol)
    qid, question = next(iter(case["request"]["questions"].items()))
    labels = list(question["criteria"])
    index = payload["choice_index"]
    if (
        payload["kind"] != "choice"
        or payload["effective_kind"] != "choice"
        or payload["options"] != request["options"]
        or type(index) is not int
        or not 0 <= index < len(labels)
        or payload["choice"] != request["options"][index]
        or len(payload["probabilities"]) != len(labels)
    ):
        raise ValueError("JEV-VL decision does not match the request")
    return read_answers(
        case,
        {
            "answers": {
                qid: {
                    "type": question["type"],
                    "choice": labels[index],
                    "probabilities": dict(zip(labels, payload["probabilities"])),
                }
            }
        },
        score_decimals,
        validation_protocol,
    )


async def request_one(
    client,
    endpoint,
    case,
    model,
    timeout,
    protocol="systemone",
    score_decimals=4,
    validation_protocol=VALIDATION_PROTOCOL,
):
    started = time.perf_counter()
    record = {
        "id": case["id"],
        "status": None,
        "error": None,
        "error_kind": None,
        "response": None,
        "response_bytes_base64": None,
        "answers": {},
        "state": "completed",
    }
    body = None
    try:
        async with asyncio.timeout(timeout):
            response = await client.post(
                endpoint, json=wire_request(case, model, protocol)
            )
        record["status"] = response.status_code
        body = response.content
        record["response_bytes_base64"] = base64.b64encode(body).decode("ascii")
        record["response"] = body.decode("utf-8", errors="replace")
        response.raise_for_status()
        record["answers"] = wire_answers(
            case,
            response_payload(body),
            model,
            protocol,
            score_decimals,
            validation_protocol,
        )
    except httpx.HTTPStatusError as exc:
        record.update(error=str(exc), error_kind=f"http_{record['status']}")
    except (TimeoutError, httpx.TimeoutException) as exc:
        record.update(
            error=str(exc) or "request deadline exceeded", error_kind="timeout"
        )
    except httpx.RequestError as exc:
        record.update(error=str(exc), error_kind="transport")
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        record.update(error=str(exc), error_kind="invalid_response")
    except asyncio.CancelledError:
        record.update(
            state="incomplete",
            error="request cancelled before response validation completed",
            error_kind="interrupted",
        )
    duration = (time.perf_counter() - started) * 1000
    record["latency_ms"] = duration if record["state"] == "completed" else None
    if record["state"] == "incomplete":
        record["elapsed_until_abort_ms"] = duration
    record["response_sha256"] = (
        hashlib.sha256(body).hexdigest() if body is not None else None
    )
    return record


def measurement_state(output):
    state = {
        "started_at": None,
        "finished_at": None,
        "wall_seconds": None,
        "stop_reason": None,
        "attempted_ids": [],
        "completed_ids": [],
        "active_ids": [],
        "responses_sha256": None,
    }
    if output is not None:
        state["manifest_sha256"] = digest(output / "requests.jsonl")
        state["config_sha256"] = digest(output / "config.json")
        state["warmup_sha256"] = digest(output / "warmup.json")
    return state


async def replay(
    client,
    endpoint,
    cases,
    model,
    concurrency,
    timeout,
    output=None,
    *,
    protocol="systemone",
    score_decimals=4,
    validation_protocol=VALIDATION_PROTOCOL,
):
    pending = iter(enumerate(cases))
    records = [None] * len(cases)
    state = measurement_state(output)
    responses = None
    response_hash = hashlib.sha256()
    if output is not None:
        state["responses_sha256"] = response_hash.hexdigest()
        responses = (output / "responses.jsonl").open("w", encoding="utf-8", newline="")

    def save_state():
        if output is not None:
            write_json(output / "completion.json", state)

    async def worker():
        for index, case in pending:
            state["attempted_ids"].append(case["id"])
            state["active_ids"].append(case["id"])
            save_state()
            records[index] = await request_one(
                client,
                endpoint,
                case,
                model,
                timeout,
                protocol,
                score_decimals,
                validation_protocol,
            )
            record = records[index]
            if responses is not None:
                line = json.dumps(record, allow_nan=False) + "\n"
                responses.write(line)
                responses.flush()
                response_hash.update(line.encode("utf-8"))
                state["responses_sha256"] = response_hash.hexdigest()
            if record["state"] == "completed":
                state["completed_ids"].append(case["id"])
            state["active_ids"].remove(case["id"])
            save_state()
            if record["state"] == "incomplete":
                raise asyncio.CancelledError

    started = time.perf_counter()
    state["started_at"] = time.time()
    save_state()
    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
        if responses is not None:
            responses.close()
            responses = None
            path = output / "responses.jsonl"
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "".join(json.dumps(r, allow_nan=False) + "\n" for r in records),
                encoding="utf-8",
                newline="",
            )
            temporary.replace(path)
            state["responses_sha256"] = digest(path)
        state["stop_reason"] = "completed"
    except BaseException as exc:
        state["stop_reason"] = (
            "interrupted" if isinstance(exc, asyncio.CancelledError) else "error"
        )
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        state["wall_seconds"] = time.perf_counter() - started
        state["finished_at"] = time.time()
        if responses is not None:
            responses.close()
        save_state()
    return records, state["wall_seconds"]


async def run(args, cases):
    protocol, score_decimals, version = validation_settings(
        {
            "protocol": getattr(args, "protocol", "systemone"),
            "score_decimals": getattr(args, "score_decimals", 4),
            "validation_protocol": VALIDATION_PROTOCOL,
        }
    )
    metadata = json.loads(args.metadata.read_text())
    for name in (
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
    ):
        if not metadata.get(name):
            raise ValueError(f"metadata requires {name}")
    _refuse_numeric_screen_baseline(args)
    fast_screen = getattr(args, "fast_screen", False)
    baseline_screen = getattr(args, "baseline_screen", None)
    screen_record = None
    if fast_screen:
        _require_fast_screen_shape(args)
        cases, bucket_counts, quota_shortfall = fast_screen_sample(cases)
        selected_ids = [case["id"] for case in cases]
        parent_sha = digest(args.manifest)
        screen_record = {
            "seed": FAST_SCREEN_SEED,
            "relative_worse": FAST_SCREEN_RELATIVE,
            "absolute_worse_ms": FAST_SCREEN_ABSOLUTE_MS,
            "buckets": [
                {
                    "name": name,
                    "min_tokens": lo,
                    "max_tokens_exclusive": hi,
                    "quota": quota,
                }
                for name, lo, hi, quota in FAST_SCREEN_BUCKETS
            ],
            "quotas": fast_screen_quotas(),
            "selected_ids": selected_ids,
            "bucket_counts": bucket_counts,
            "quota_shortfall": quota_shortfall,
            "parent_manifest_sha256": parent_sha,
        }
        if baseline_screen is not None:
            screen_record["role"] = "tip"
            screen_record.update(
                load_same_request_screen_baseline(
                    baseline_screen,
                    selected_ids,
                    parent_sha,
                    metadata.get("gpu"),
                )
            )
        else:
            screen_record["role"] = "screen_primary"
    elif baseline_screen is not None:
        raise ValueError("--baseline-screen requires --fast-screen")
    warmup_manifest = getattr(args, "warmup_manifest", None)
    warmup_cases = load_cases(warmup_manifest) if warmup_manifest else cases
    # A case the adapter cannot wire fails before any HTTP exchange; reject it
    # before creating the output directory so a saved archive stays summarizable.
    for case in cases if warmup_cases is cases else [*cases, *warmup_cases]:
        try:
            wire_request(case, args.model, protocol)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"case {case['id']!r} is incompatible with the {protocol} "
                f"wire protocol: {exc}"
            ) from exc
    if fast_screen:
        request_bytes = manifest_subset_bytes(args.manifest, selected_ids)
    else:
        request_bytes = args.manifest.read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "manifest_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "runner_sha256": digest(Path(__file__)),
        "metadata": metadata,
        "validation_protocol": version,
        "score_decimals": score_decimals,
        "protocol": protocol,
        "response_format": RESPONSE_FORMAT,
        "endpoint": args.endpoint,
        "model": args.model,
        "phase": args.phase,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "warmup_manifest_sha256": digest(warmup_manifest) if warmup_manifest else None,
        "timeout_seconds": args.timeout,
        "latency_basis": "post_to_body_json_and_schema_validation",
        "python": platform.python_version(),
        "httpx": httpx.__version__,
    }
    if screen_record is not None:
        config["fast_screen"] = screen_record
    write_json(args.output / "config.json", config)
    (args.output / "requests.jsonl").write_bytes(request_bytes)
    if warmup_manifest:
        (args.output / "warmup-requests.jsonl").write_bytes(
            warmup_manifest.read_bytes()
        )
    headers = {}
    if token := os.environ.get("OMNI_JEV_TEST_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(
        headers=headers,
        trust_env=False,
        timeout=None,
        limits=httpx.Limits(
            max_connections=args.concurrency, max_keepalive_connections=args.concurrency
        ),
    ) as client:
        # The first real inference validates readiness; preparation is not timed as load.
        warmup = []
        for i in range(args.warmup):
            warmup.append(
                await request_one(
                    client,
                    args.endpoint,
                    warmup_cases[i % len(warmup_cases)],
                    args.model,
                    args.timeout,
                    protocol,
                    score_decimals,
                    version,
                )
            )
            if warmup[-1]["error"] is not None:
                write_json(args.output / "warmup.json", warmup)
                state = measurement_state(args.output)
                state["stop_reason"] = (
                    "warmup_interrupted"
                    if warmup[-1]["state"] == "incomplete"
                    else "warmup_failed"
                )
                write_json(args.output / "completion.json", state)
                write_json(args.output / "summary.json", summarize_saved(args.output))
                raise ValueError("readiness/warmup failed; see warmup.json")
        write_json(args.output / "warmup.json", warmup)
        try:
            await replay(
                client,
                args.endpoint,
                cases,
                args.model,
                args.concurrency,
                args.timeout,
                args.output,
                protocol=protocol,
                score_decimals=score_decimals,
                validation_protocol=version,
            )
        except asyncio.CancelledError:
            # Replay has saved the completed samples and the measured stop boundary.
            pass
    summary = summarize_saved(args.output)
    if summary.get("fast_screen_decision") == "drop" and "dropped_at" not in summary:
        saved_screen = json.loads((args.output / "config.json").read_text())["fast_screen"]
        write_json(
            args.output / FAST_SCREEN_DROP_NAME,
            {
                "decision": "drop",
                "dropped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "selected_ids": saved_screen["selected_ids"],
                "baseline_runtime_revision": saved_screen["baseline_runtime_revision"],
                "baseline_p50_ms": saved_screen["baseline_p50_ms"],
                "baseline_p95_ms": saved_screen["baseline_p95_ms"],
                "tip_p50_ms": summary["successful_latency_p50_ms"],
                "tip_p95_ms": summary["successful_latency_p95_ms"],
            },
        )
        summary = summarize_saved(args.output)
        if "dropped_at" not in summary:
            raise ValueError("fast-screen DROP did not record dropped_at")
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return int(summary["failed_requests"] > 0 or not summary["evidence_complete"])


def summarize_saved(path):
    """Recompute from this runner's saved inputs, responses and timing evidence."""
    config = json.loads((path / "config.json").read_text())
    state = json.loads((path / "completion.json").read_text())
    if digest(path / "config.json") != state["config_sha256"]:
        raise ValueError("saved config checksum does not match completion")
    if "warmup_sha256" not in state:
        raise ValueError(
            "saved warmup checksum missing; use the collector that produced this run"
        )
    if (
        not (path / "warmup.json").is_file()
        or digest(path / "warmup.json") != state["warmup_sha256"]
    ):
        raise ValueError("saved warmup checksum does not match completion")
    protocol, score_decimals, version = validation_settings(config)
    response_format = saved_response_format(config)
    if config["latency_basis"] != "post_to_body_json_and_schema_validation":
        raise ValueError("unsupported saved latency basis")
    manifest_hash = digest(path / "requests.jsonl")
    if (
        manifest_hash != config["manifest_sha256"]
        or manifest_hash != state["manifest_sha256"]
    ):
        raise ValueError("saved manifest checksum does not match config/completion")
    if config["warmup_manifest_sha256"] is not None:
        if digest(path / "warmup-requests.jsonl") != config["warmup_manifest_sha256"]:
            raise ValueError("saved warmup manifest checksum does not match config")
    responses_path = path / "responses.jsonl"
    if state["started_at"] is None:
        if state["responses_sha256"] is not None or responses_path.exists():
            raise ValueError("unstarted measurement must not claim responses")
        records = []
    elif digest(responses_path) != state["responses_sha256"]:
        raise ValueError("saved responses checksum does not match completion")
    else:
        records = [json.loads(line) for line in responses_path.read_text().splitlines()]
    cases = load_cases(path / "requests.jsonl")
    screen = config.get("fast_screen")
    if screen is not None:
        _check_saved_fast_screen(screen, cases, config)
    by_id = {c["id"]: c for c in cases}
    ids = {}
    for name in ("attempted_ids", "completed_ids", "active_ids"):
        values = state[name]
        if len(values) != len(set(values)) or not set(values) <= set(by_id):
            raise ValueError(f"duplicate or unknown {name}")
        ids[name] = set(values)
    if not ids["completed_ids"] <= ids["attempted_ids"]:
        raise ValueError("completed IDs must have a saved start")
    if not ids["active_ids"] <= ids["attempted_ids"] - ids["completed_ids"]:
        raise ValueError("active IDs must be attempted and incomplete")
    record_ids = [r["id"] for r in records]
    if (
        len(record_ids) != len(set(record_ids))
        or not set(record_ids) <= ids["attempted_ids"]
    ):
        raise ValueError("duplicate, unknown or unattempted response ID")
    completed_ids = set()
    for record in records:
        body = response_body(record, response_format)
        if record["state"] == "incomplete":
            if (
                record["latency_ms"] is not None
                or record["error_kind"] != "interrupted"
                or not record["error"]
                or record["answers"]
            ):
                raise ValueError("incomplete response must not claim a latency/answer")
            if (
                not number(record["elapsed_until_abort_ms"])
                or record["elapsed_until_abort_ms"] < 0
            ):
                raise ValueError("invalid elapsed-until-abort")
            continue
        if (
            record["state"] != "completed"
            or not number(record["latency_ms"])
            or record["latency_ms"] < 0
        ):
            raise ValueError("invalid completed response or latency")
        completed_ids.add(record["id"])
        answers, error_kind = {}, None
        status = record["status"]
        if status is None:
            if record["response"] is not None or record["error_kind"] not in (
                "timeout",
                "transport",
            ):
                raise ValueError("response without status must be a client failure")
            error_kind = record["error_kind"]
        elif (
            not isinstance(status, int)
            or isinstance(status, bool)
            or not 100 <= status <= 999
        ):
            raise ValueError("invalid HTTP status")
        elif not 200 <= status < 300:
            error_kind = f"http_{status}"
        else:
            try:
                answers = wire_answers(
                    by_id[record["id"]],
                    response_payload(body),
                    config.get("model"),
                    protocol,
                    score_decimals,
                    version,
                )
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
                error_kind = "invalid_response"
        if (
            record["answers"] != answers
            or record["error_kind"] != error_kind
            or (record["error"] is None) != (error_kind is None)
        ):
            raise ValueError("raw response and saved validation result disagree")
    if completed_ids != ids["completed_ids"]:
        raise ValueError("completed IDs do not match saved responses")
    elapsed = state["wall_seconds"]
    if state["started_at"] is None:
        if (
            state["finished_at"] is not None
            or elapsed is not None
            or any(ids.values())
            or state["stop_reason"] not in ("warmup_failed", "warmup_interrupted")
        ):
            raise ValueError("invalid unstarted measurement")
    elif not number(state["started_at"]):
        raise ValueError("invalid measured start boundary")
    elif state["finished_at"] is None:
        if elapsed is not None or state["stop_reason"] is not None:
            raise ValueError("unfinished measurement must not claim a terminal wall")
    else:
        if (
            not number(state["finished_at"])
            or state["finished_at"] < state["started_at"]
            or not number(elapsed)
            or elapsed <= 0
            or ids["active_ids"]
            or state["stop_reason"] not in ("completed", "interrupted", "error")
        ):
            raise ValueError("invalid measured completion boundary")
        if state["stop_reason"] == "completed" and completed_ids != set(by_id):
            raise ValueError("completed traversal is missing response IDs")
    summary = summarize(cases, records, elapsed, ids["attempted_ids"])
    summary["stop_reason"] = state["stop_reason"]
    summary["evidence_complete"] = (
        summary["evidence_complete"] and state["stop_reason"] == "completed"
    )
    if screen is not None:
        summary["fast_screen_decision"] = _saved_fast_screen_decision(screen, summary)
        summary["quota_shortfall"] = screen.get("quota_shortfall") or []
        if screen.get("role") == "tip":
            p50 = summary["successful_latency_p50_ms"]
            p95 = summary["successful_latency_p95_ms"]
            summary["fast_screen_delta_p50_ms"] = (
                None if p50 is None else p50 - screen["baseline_p50_ms"]
            )
            summary["fast_screen_delta_p95_ms"] = (
                None if p95 is None else p95 - screen["baseline_p95_ms"]
            )
        _attach_dropped_at(path, screen, summary)
    elif (path / FAST_SCREEN_DROP_NAME).exists():
        raise ValueError("fast_screen_drop.json is only written for a DROP")
    return summary


def compare(args):
    configs = [
        json.loads((p / "config.json").read_text())
        for p in (args.reference, args.candidate)
    ]
    for path, config in zip((args.reference, args.candidate), configs):
        validation_settings(config)
        saved_response_format(config)
        if digest(path / "requests.jsonl") != config["manifest_sha256"]:
            raise ValueError("saved manifest checksum does not match config")
    if configs[0]["manifest_sha256"] != configs[1]["manifest_sha256"]:
        raise ValueError("cannot compare different request manifests")
    runs = [
        [json.loads(line) for line in (p / "responses.jsonl").read_text().splitlines()]
        for p in (args.reference, args.candidate)
    ]
    drift, flips, score_drift, failed, compared = 0.0, 0, 0.0, 0, 0
    cases = load_cases(args.reference / "requests.jsonl")
    ordered = []
    for records in runs:
        by_id = {r["id"]: r for r in records}
        if len(by_id) != len(records) or set(by_id) != {c["id"] for c in cases}:
            raise ValueError("response IDs do not uniquely cover the manifest")
        ordered.append([by_id[c["id"]] for c in cases])
    for case, ref, candidate in zip(cases, *ordered):
        if ref["error"] or candidate["error"]:
            failed += 1
            continue
        for qid, question in case["request"]["questions"].items():
            a, b = ref["answers"][qid], candidate["answers"][qid]
            drift = max(
                drift,
                max(
                    abs(p - b["probabilities"][k])
                    for k, p in a["probabilities"].items()
                ),
            )
            kind = question["type"]
            if kind == "score":
                score_drift = max(score_drift, abs(a["value"] - b["value"]))
            elif kind == "noul":
                flips += (a["value"] >= 0.5) != (b["value"] >= 0.5)
            else:
                flips += a["value"] != b["value"]
            compared += 1
    passed = (
        failed == 0
        and compared > 0
        and drift <= args.max_probability_drift
        and score_drift <= args.max_score_drift
        and flips <= args.max_flips
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "compared_decisions": compared,
                "failed_request_pairs": failed,
                "decision_flips": flips,
                "max_probability_drift": drift,
                "max_score_drift": score_drift,
            },
            indent=2,
        )
    )
    return int(not passed)


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    runner = commands.add_parser("run")
    runner.add_argument("manifest", type=Path)
    runner.add_argument("--endpoint", required=True, help="full /v1/systemone URL")
    runner.add_argument(
        "--model", required=True, help="backend's alias for the pinned checkpoint"
    )
    runner.add_argument(
        "--protocol", choices=("systemone", "jev-vl"), default="systemone"
    )
    runner.add_argument("--score-decimals", type=int, choices=(2, 3, 4), default=4)
    runner.add_argument("--metadata", type=Path, required=True)
    runner.add_argument(
        "--output", type=Path, required=True, help="new directory; never overwritten"
    )
    runner.add_argument("--phase", choices=("feasibility", "measured"), required=True)
    runner.add_argument("--concurrency", type=positive, default=1)
    runner.add_argument("--warmup", type=positive, default=3)
    runner.add_argument(
        "--fast-screen",
        action="store_true",
        help=(
            "sample Public231 requests (seed 231, quotas 10/10/8/8/6/16) for "
            "one measured round. A short bucket is capped and recorded. "
            "Without --baseline-screen this records the screen primary. "
            "A tip passes that directory, not pooled C1 numbers."
        ),
    )
    runner.add_argument(
        "--baseline-screen",
        type=Path,
        help=(
            "primary SHA's own fast-screen directory of these same request IDs. "
            "Not a full-C1 pooled p50/p95."
        ),
    )
    runner.add_argument(
        "--baseline-p50-ms",
        type=float,
        help=argparse.SUPPRESS,
    )
    runner.add_argument(
        "--baseline-p95-ms",
        type=float,
        help=argparse.SUPPRESS,
    )
    runner.add_argument(
        "--warmup-manifest", type=Path, help="independent labelled warmup requests"
    )
    runner.add_argument("--timeout", type=positive, default=60)
    comparison = commands.add_parser("compare")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--max-probability-drift", type=float, required=True)
    comparison.add_argument("--max-score-drift", type=float, required=True)
    comparison.add_argument("--max-flips", type=int, required=True)
    summary = commands.add_parser(
        "summary", help="recompute a saved run without HTTP requests"
    )
    summary.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "summary":
            print(json.dumps(summarize_saved(args.output), indent=2, allow_nan=False))
            return 0
        if args.command == "compare":
            tolerances = (
                args.max_probability_drift,
                args.max_score_drift,
                args.max_flips,
            )
            if any(not number(v) or v < 0 for v in tolerances):
                raise ValueError("comparison tolerances must be finite and nonnegative")
            return compare(args)
        cases = load_cases(args.manifest)
        if args.command == "validate":
            print(f"Validated {len(cases)} requests; SHA256 {digest(args.manifest)}")
            return 0
        return asyncio.run(run(args, cases))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
