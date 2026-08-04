# Nudge — Deployment

Real-time CATE scoring via NVIDIA Triton Inference Server on AWS SageMaker.

## Architecture

```
Customer features (JSON)
        ↓
inference_client.py  — feature engineering (mirrors queries/02_hillstrom_features.sql)
        ↓
SageMaker endpoint  —  NVIDIA Triton + RAPIDS FIL backend
        ↓
CATE score + persuadability label + action recommendation
```

## Instance options

| Mode | Instance | Cost | Use case |
|---|---|---|---|
| GPU | ml.g4dn.xlarge | ~$0.74/hr | Production, batch scoring |
| CPU | ml.c5.large | ~$0.10/hr | Demo, low-traffic |

## Quickstart

```bash
# 1. Export model artifacts from Phase 4 notebook output
python deployment/export_model.py --model-pkl outputs/phase4_model.pkl

# 2. Deploy (GPU, persistent)
python deployment/sagemaker/deploy_sagemaker.py --bucket my-bucket

# 2b. Demo mode (deploys then immediately tears down — no ongoing cost)
python deployment/sagemaker/deploy_sagemaker.py --bucket my-bucket --demo

# 3. Score a customer
python deployment/sagemaker/inference_client.py

# 4. Set up CloudWatch monitoring
python deployment/sagemaker/monitoring_setup.py

# 5. Tear down when done
python deployment/sagemaker/deploy_sagemaker.py --teardown
```

## Triton model repository

```
deployment/model_artifacts/
  nudge_cate_model.json   ← written by notebooks/phase4_dr_learner_robustness.ipynb
  model_metadata.json     ← feature list, fidelity check, CATE distribution
deployment/triton_model_repo/
  nudge_cate_model/
    1/
      xgboost.json        ← staged by export_model.py (gitignored)
    config.pbtxt          ← FIL backend config, KIND_GPU
deployment/triton/
  config_cpu.pbtxt        ← KIND_CPU variant
```

The GPU `config.pbtxt` inside the model repository is canonical. For a CPU
instance, `deploy_sagemaker.py` copies the repository to a temp directory and
swaps `config_cpu.pbtxt` in there, so a CPU deploy never leaves `KIND_CPU`
behind in the tracked config.

`export_model.py` refuses to stage an artifact whose feature count disagrees
with `config.pbtxt`, or whose Spearman fidelity against the DR-Learner's own
`effect()` output is below 0.9999.

## Prerequisites

- AWS credentials configured (`aws configure`)
- SageMaker execution role with S3 and ECR access
- `pip install -r requirements-deploy.txt` (adds `boto3` and `sagemaker`)

## Cost note

`deploy_sagemaker.py --demo` deploys, runs a smoke test, then deletes the endpoint.
Total cost: < $0.02. Use this for portfolio demos and CI/CD validation.
