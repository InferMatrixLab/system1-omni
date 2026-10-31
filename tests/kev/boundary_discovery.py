"""Freeze deduplicated development Noul inputs before official-only discovery."""
import argparse
import collections
import json
from pathlib import Path

from service_acceptance import GATES, REFERENCE, ROOT, digest, module, save


def source_identity(case, question_id):
    if "origin" in case:
        return tuple(case["origin"][k] for k in ("record_id", "question_id"))
    # Existing acceptance names are suite:source-record-id; synthetic names have no colon.
    if ":" in case["name"]:
        return case["name"].split(":", 1)[1], question_id
    return None


def previous_keys(corpora):
    identities, tokens = set(), set()
    for corpus in corpora:
        if corpus["gates"] != GATES:
            raise ValueError("previous corpus gates changed")
        for case in corpus["cases"]:
            for meta, row in zip(case["meta"], case["rows"], strict=True):
                if meta["type"] == "noul":
                    key = source_identity(case, meta["id"])
                    if key:
                        identities.add(key)
                    tokens.add(tuple(row["ids"]))
    return identities, tokens


def prepare(args):
    if args.out.exists():
        raise ValueError("preserve previous discovery corpus")
    args.out.mkdir(parents=True)
    pins = json.loads((ROOT / "tests/kev/data/boundary-sources.json").read_text())
    if pins["reference"] != REFERENCE:
        raise ValueError("reference pin mismatch")
    for name, sha in pins["sources"].items():
        if digest(args.reference / name) != sha:
            raise ValueError(f"changed development source: {name}")
    fixture = module("kev_fixture", ROOT / "tests/kev/generate_processing_fixture.py")
    api, model = fixture.official(args.reference)
    tok = model.load_tokenizer(args.base)
    seen, token_rows = previous_keys([json.loads(p.read_text()) for p in args.previous])
    captured, exclusions = [], []
    for source in pins["sources"]:
        records = [json.loads(line) for line in (args.reference / source).read_text().splitlines() if line]
        for record in records:
            meta = record["_meta"]
            if meta.get("variant") != "clean":
                continue
            for qid, question in record["questions"].items():
                if question["type"] != "noul":
                    continue
                origin = {"file": source, "record_id": meta["id"], "question_id": qid,
                          "source": meta["source"], "group_id": meta.get("group_id", meta["id"])}
                identity = meta["id"], qid
                if identity in seen:
                    exclusions.append({**origin, "reason": "existing_source_question"})
                    continue
                seen.add(identity)
                request = {"model": "kev-latest", "state": record["state"], "questions": {
                    qid: {k: v for k, v in question.items() if k in ("type", "instructions", "criteria")}}}
                case = {"name": f"noul-v4:{source}:{meta['id']}:{qid}", "population": "discovery",
                        "source": meta["source"], "group": meta.get("group_id", meta["id"]),
                        "labels": {}, "source_label": question.get("label"), "origin": origin, "request": request}
                # Strict encoding does not truncate. Length exclusions happen before any model run.
                try:
                    item = fixture.capture(case, api, model, tok)
                except model.ContextOverflow as error:
                    exclusions.append({**origin, "reason": "encoder_context_limit", "detail": str(error)})
                    continue
                ids = tuple(item["rows"][0]["ids"])
                split = ids.index(248061)
                if split > 1024 or len(ids)-split > 1024:
                    exclusions.append({**origin, "reason": "discovery_length_budget", "state": split, "branch": len(ids)-split})
                    continue
                if ids in token_rows:
                    exclusions.append({**origin, "reason": "identical_token_row"})
                    continue
                token_rows.add(ids)
                # capture() includes artificial CPU probabilities for postprocess tests only;
                # remove those from the discovery corpus so membership cannot use them.
                for key in ("probabilities", "answers", "output_tokens"):
                    item.pop(key)
                captured.append(item)
    if not captured or len({c["name"] for c in captured}) != len(captured):
        raise ValueError("empty or duplicate discovery corpus")
    corpus = {"reference": REFERENCE, "sources": pins["sources"], "gates": GATES,
              "counts": {"discovery": len(captured)}, "cases": captured}
    save(args.out / "corpus.json", corpus)
    save(args.out / "exclusions.json", exclusions)
    save(args.out / "freeze.json", {
        "generator_sha256": digest(__file__), "source_manifest_sha256": digest(ROOT / "tests/kev/data/boundary-sources.json"),
        "corpus_sha256": digest(args.out / "corpus.json"), "requests": len(captured),
        "previous": {str(p): digest(p) for p in args.previous},
        "source_counts": collections.Counter(c["source"] for c in captured),
        "exclusion_counts": collections.Counter(e["reason"] for e in exclusions)})
    print(json.dumps(json.loads((args.out / "freeze.json").read_text()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference", "base", "out"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--previous", type=Path, action="append", required=True)
    prepare(parser.parse_args())
