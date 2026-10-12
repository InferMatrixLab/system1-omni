"""Records the hosted pplx-decider-v1.1-27b's answers to the fixture requests, for the future executor's parity
check. Calls OpenRouter's decisions endpoint (model perplexity/pplx-decider-v1.1-27b) once per case.

Usage: OPENROUTER_API_KEY=... python tests/pplx_decider/record_hosted.py tests/pplx_decider/data
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(sys.argv[1])
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "perplexity/pplx-decider-v1.1-27b"
fixture = json.loads((DATA / "contract.json").read_text(encoding="utf-8"))
records = []
for case in fixture["cases"]:
    body = dict(case["request"], model=MODEL)
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:  # the hosted API's own validation, kept as evidence
        detail = json.loads(error.read() or b"null")
        if isinstance(detail, dict):
            detail.pop("user_id", None)  # the caller's account id; not evidence
        records.append({"name": case["name"], "status": error.code, "error": detail})
        continue
    records.append(
        {
            "name": case["name"],
            "model": payload.get("model"),
            "answers": payload["answers"],
            "input_tokens": payload.get("usage", {}).get("input_tokens"),
        }
    )
(DATA / "hosted.json").write_text(
    json.dumps({"endpoint": URL, "requested_model": MODEL, "cases": records}, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(len(records), "cases recorded")
