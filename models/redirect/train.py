#!/usr/bin/env python3
"""Trainer skeleton for the redirect model.

Invoked by `cradle train redirect` as: `uv run python train.py`. cradle
sets three env vars:

    CRADLE_MODEL_NAME       — "redirect"
    CRADLE_DATA_DIR         — absolute path to data/ (train/val/test.jsonl)
    CRADLE_OUTPUT_DIR       — absolute path to this model dir; write
                              checkpoint.safetensors + metrics.json here.

v0.1 of cradle does *not* ship this script — it's the user's responsibility
to fill in. This stub exists so `cradle build redirect` exercises the
shellout path without errors. Replace the body with real PyTorch
training as part of the follow-on PRD-cradle-bake-integration work.
"""

import json
import os
import pathlib
import sys


def main() -> int:
    model = os.environ.get("CRADLE_MODEL_NAME", "?")
    data = pathlib.Path(os.environ.get("CRADLE_DATA_DIR", ""))
    out = pathlib.Path(os.environ.get("CRADLE_OUTPUT_DIR", "."))
    out.mkdir(parents=True, exist_ok=True)
    # Emit a placeholder metrics.json so the receipt step in the
    # follow-on PRD has *something* to assert against; real values
    # come from actual training.
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "cradle.metrics.v1",
                "model": model,
                "note": "stub — replace with real training output",
                "data_dir": str(data),
                "test_accuracy": None,
                "test_auc": None,
            },
            indent=2,
        )
    )
    print(f"cradle train stub: wrote {out/'metrics.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
