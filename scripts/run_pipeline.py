"""
Execute the Nudge notebook pipeline in dependency order.

Phases must run 1 -> 2 -> 4 -> 5a: each reads the previous phase's parquet from
outputs/. Running them out of order silently mixes results from different runs,
which is how the repo previously ended up with a comparison table that disagreed
with the notebooks that produced it.

Usage:
    python scripts/run_pipeline.py              # all phases
    python scripts/run_pipeline.py 4 5a         # just these
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
NOTEBOOKS = ROOT / 'notebooks'

PHASES = {
    '1': 'phase1_naive_baseline.ipynb',
    '2': 'phase2_meta_learners.ipynb',
    '4': 'phase4_dr_learner_robustness.ipynb',
    '5a': 'phase5a_budget_optimizer.ipynb',
}


def repair_outputs(path: Path) -> None:
    """
    Restore required nbformat fields on stored outputs.

    Editors that write notebooks programmatically routinely drop `name` from
    stream outputs and `metadata` from execute_result, which makes nbconvert
    refuse the file before it runs a single cell.
    """
    nb = json.loads(path.read_text(encoding='utf-8'))
    fixed = 0
    for cell in nb.get('cells', []):
        for out in cell.get('outputs', []):
            kind = out.get('output_type')
            if kind == 'stream' and 'name' not in out:
                out['name'] = 'stdout'
                fixed += 1
            if kind in ('execute_result', 'display_data') and 'metadata' not in out:
                out['metadata'] = {}
                fixed += 1
            if kind == 'execute_result' and 'execution_count' not in out:
                out['execution_count'] = None
                fixed += 1
    if fixed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
        print(f'  repaired {fixed} malformed output fields')


def run(phase: str) -> bool:
    path = NOTEBOOKS / PHASES[phase]
    print(f'\n=== Phase {phase}: {path.name} ===')
    repair_outputs(path)
    result = subprocess.run(
        [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook',
         '--execute', '--inplace', '--ExecutePreprocessor.timeout=5400', str(path)],
        cwd=ROOT,
    )
    ok = result.returncode == 0
    print(f'  {"OK" if ok else "FAILED"}')
    return ok


def main() -> int:
    requested = sys.argv[1:] or list(PHASES)
    unknown = [p for p in requested if p not in PHASES]
    if unknown:
        print(f'Unknown phase(s): {unknown}. Choose from {list(PHASES)}')
        return 2

    for phase in requested:
        if not run(phase):
            print(f'\nStopped at phase {phase}. Later phases would consume stale outputs.')
            return 1

    print('\nPipeline complete.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
