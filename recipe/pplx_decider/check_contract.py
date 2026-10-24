"""Compare the CPU port with the pinned checkpoint's own functions.

Run from the repository root. This does not load weights.

    PYTHONPATH=src python3 recipe/pplx_decider/check_contract.py /path/to/checkpoint
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.pplx_decider.contract import answer, messages  # noqa: E402
from models.pplx_decider.engine import verify_checkpoint  # noqa: E402


def reference_functions(model_py: Path):
    source = model_py.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = ["describe", "options", "decision_messages", "answer"]
    chunks = [
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    text = (
        "from __future__ import annotations\nimport json, math\n"
        "from collections.abc import Sequence\nMAX_OPTIONS = 255\n"
        + "\n\n".join(chunks)
        + "\n"
    )
    namespace: dict = {}
    exec(compile(text, "model.py", "exec"), namespace)
    return namespace


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_contract.py CHECKPOINT_DIR", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    verify_checkpoint(root)
    functions = reference_functions(root / "source/src/autojev/model.py")
    fixture = json.loads(
        (ROOT / "tests/pplx_decider/fixtures/reference_cases.json").read_text(encoding="utf-8")
    )
    for case in fixture["cases"]:
        row = {
            "state": case["state"],
            "question": case["question"],
            "images": [None] * case["n_images"],
        }
        config = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
        reference_messages = functions["decision_messages"](row, config["codes"])
        if reference_messages != messages(case["state"], case["question"], case["n_images"]):
            raise SystemExit(f"prompt mismatch: {case['name']}")
        if "answer" not in case:
            continue
        reference_answer = functions["answer"](case["question"], case["probabilities_in"])
        if reference_answer != answer(case["question"], case["probabilities_in"]):
            raise SystemExit(f"answer mismatch: {case['name']}")
        if reference_answer != case["answer"]:
            raise SystemExit(f"fixture mismatch: {case['name']}")
    print(f"matched {len(fixture['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
