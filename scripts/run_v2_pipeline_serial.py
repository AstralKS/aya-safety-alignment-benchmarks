"""Launch the v2 coordinator with both scoring entrypoints forced to concurrency 1."""
from __future__ import annotations

import functools
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_baseline_pipeline as pipeline
import v2_glm_scoring as scoring

if __name__ == '__main__':
    pipeline.scoring.score_fixed = functools.partial(scoring.score_fixed, concurrency=1)
    pipeline.scoring.factscore_stage = functools.partial(scoring.factscore_stage, concurrency=1)
    interval = 60
    result = pipeline.run(Path(__file__).resolve().parents[1], interval)
    print('pipeline finished: complete=%s' % result.get('complete'))
