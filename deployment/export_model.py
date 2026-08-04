"""
Package the trained CATE model into a Triton model repository.

Phase 4 writes the serving artifact directly:

    deployment/model_artifacts/nudge_cate_model.json   XGBoost final stage
    deployment/model_artifacts/model_metadata.json     feature schema + score stats

The model is the DR-Learner's final stage — the only learner in this project whose
CATE is a single `X -> tau(x)` map, and therefore the only one that fits a
single-tensor FIL signature. Phase 4 asserts the exported model reproduces
`DRLearner.effect()` exactly before writing it.

This script validates that artifact against config.pbtxt and stages it into the
versioned model repository that deploy_sagemaker.py tarballs.

Usage:
    python deployment/export_model.py
    python deployment/export_model.py --version 2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

DEPLOYMENT = Path(__file__).parent
ARTIFACTS_DIR = DEPLOYMENT / 'model_artifacts'
MODEL_REPO = DEPLOYMENT / 'triton_model_repo'
MODEL_NAME = 'nudge_cate_model'

# FIL resolves the weights file by name; xgboost_json expects xgboost.json.
TRITON_MODEL_FILENAME = 'xgboost.json'


def config_input_dims(config_path: Path) -> int:
    """Input width declared in config.pbtxt, so we can catch schema drift."""
    text = config_path.read_text()
    match = re.search(r'input\s*\[.*?dims:\s*\[\s*(\d+)\s*\]', text, re.S)
    if not match:
        raise ValueError(f'Could not parse input dims from {config_path}')
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=1,
                        help='Triton model version directory to write into')
    args = parser.parse_args()

    model_src = ARTIFACTS_DIR / f'{MODEL_NAME}.json'
    meta_src = ARTIFACTS_DIR / 'model_metadata.json'

    if not model_src.exists() or not meta_src.exists():
        print(f'Missing artifacts in {ARTIFACTS_DIR}.')
        print('Run notebooks/phase4_dr_learner_robustness.ipynb to generate '
              f'{MODEL_NAME}.json and model_metadata.json.')
        return 1

    metadata = json.loads(meta_src.read_text())
    config_path = MODEL_REPO / MODEL_NAME / 'config.pbtxt'
    declared = config_input_dims(config_path)

    if metadata['n_features'] != declared:
        print(f'Feature count mismatch: model was trained on '
              f'{metadata["n_features"]} features but {config_path.name} '
              f'declares {declared}. Update config.pbtxt before deploying.')
        return 1

    fidelity = metadata.get('fidelity_spearman')
    if fidelity is not None and fidelity < 0.9999:
        print(f'Artifact does not faithfully reproduce the DR-Learner '
              f'(Spearman {fidelity}). Refusing to package.')
        return 1

    version_dir = MODEL_REPO / MODEL_NAME / str(args.version)
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, version_dir / TRITON_MODEL_FILENAME)
    shutil.copy2(meta_src, MODEL_REPO / MODEL_NAME / 'model_metadata.json')

    size_kb = (version_dir / TRITON_MODEL_FILENAME).stat().st_size / 1024
    print(f'Staged {MODEL_NAME} v{args.version}')
    print(f'  {version_dir / TRITON_MODEL_FILENAME}  ({size_kb:.1f} KB)')
    print(f'  {metadata["n_features"]} features, {metadata["n_scored"]:,} customers scored')
    print(f'  CATE mean {metadata["cate_mean"]:+.6f}, std {metadata["cate_std"]:.6f}')
    print('\nNext: python deployment/sagemaker/deploy_sagemaker.py --instance-type ml.g4dn.xlarge')
    return 0


if __name__ == '__main__':
    sys.exit(main())
