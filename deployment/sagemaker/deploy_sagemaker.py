"""
Deploy Nudge CATE model to AWS SageMaker using NVIDIA Triton Inference Server.

Instance options:
  GPU  ml.g4dn.xlarge  ~$0.74/hr  — recommended for production
  CPU  ml.c5.large     ~$0.10/hr  — demo / cost-sensitive

For demo mode: deploy on demand, call predict, then delete_endpoint() to avoid
ongoing costs. See deploy() docstring.
"""

import json
import tarfile
import time
from pathlib import Path

import boto3

# `sagemaker` is imported lazily inside deploy(): packaging the model repository
# is a local, offline step and should not require the full SageMaker SDK.

# Check current: https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-triton/
TRITON_IMAGE_URI = (
    "301217895009.dkr.ecr.us-east-1.amazonaws.com/"
    "sagemaker-tritonserver:23.12-py3"
)

MODEL_NAME       = "nudge-cate-model"
ENDPOINT_NAME    = "nudge-cate-endpoint"
ENDPOINT_CONFIG  = "nudge-cate-config"


def package_model_artifacts(
    model_repo:    Path = Path('deployment/triton_model_repo'),
    output_path:   Path = Path('deployment/model.tar.gz'),
    instance_type: str = 'ml.g4dn.xlarge',
) -> Path:
    """
    Package the Triton model repository as a tar.gz for SageMaker.
    SageMaker expects model.tar.gz containing the Triton model repo at its root.

    Run deployment/export_model.py first — it stages the versioned weights and
    validates them against config.pbtxt. CPU instances get the KIND_CPU config
    swapped in, since the default config requests a GPU that is not there.

    The tarball is built from a temporary copy so that a CPU deploy never leaves
    a KIND_CPU config behind in the tracked model repository.
    """
    import shutil
    import tempfile

    weights = model_repo / 'nudge_cate_model' / '1' / 'xgboost.json'
    if not weights.exists():
        raise FileNotFoundError(
            f'{weights} not found. Run `python deployment/export_model.py` first.'
        )

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / 'model_repo'
        shutil.copytree(model_repo, staged)

        if not instance_type.startswith('ml.g'):
            cpu_config = Path('deployment/triton/config_cpu.pbtxt')
            if not cpu_config.exists():
                raise FileNotFoundError(f'{cpu_config} not found for CPU deployment.')
            shutil.copy2(cpu_config, staged / 'nudge_cate_model' / 'config.pbtxt')
            print(f'Using CPU Triton config for {instance_type}')

        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(staged, arcname='.')

    print(f'Packaged model: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)')
    return output_path


def upload_to_s3(
    local_path: Path,
    bucket: str,
    prefix: str = 'nudge/models',
) -> str:
    s3 = boto3.client('s3')
    key = f"{prefix}/{local_path.name}"
    s3.upload_file(str(local_path), bucket, key)
    s3_uri = f's3://{bucket}/{key}'
    print(f'Uploaded to {s3_uri}')
    return s3_uri


def deploy(
    bucket: str,
    region: str = 'us-east-1',
    instance_type: str = 'ml.g4dn.xlarge',
    demo_mode: bool = False,
) -> str:
    """
    Deploy Nudge CATE model as a SageMaker real-time endpoint.

    Parameters
    ----------
    bucket        : S3 bucket for model artifacts.
    region        : AWS region.
    instance_type : 'ml.g4dn.xlarge' (GPU) or 'ml.c5.large' (CPU).
    demo_mode     : If True, prints the endpoint ARN and immediately deletes it
                    to avoid ongoing costs. Set False for a persistent endpoint.

    Returns
    -------
    str — endpoint name.

    WARNING: Forgetting to teardown a persistent endpoint accrues charges.
             Run `python deploy_sagemaker.py --teardown` when done.
    """
    from sagemaker import get_execution_role

    boto_session = boto3.Session(region_name=region)
    sm_client    = boto_session.client('sagemaker')
    role         = get_execution_role()

    tar_path = package_model_artifacts(instance_type=instance_type)
    model_s3 = upload_to_s3(tar_path, bucket)

    sm_client.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            'Image': TRITON_IMAGE_URI,
            'ModelDataUrl': model_s3,
            'Environment': {
                'SAGEMAKER_TRITON_DEFAULT_MODEL_NAME': 'nudge_cate_model',
                'SAGEMAKER_TRITON_THREAD_COUNT': '4',
            },
        },
        ExecutionRoleArn=role,
    )
    print(f'Created SageMaker model: {MODEL_NAME}')

    sm_client.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG,
        ProductionVariants=[{
            'VariantName':           'AllTraffic',
            'ModelName':             MODEL_NAME,
            'InstanceType':          instance_type,
            'InitialInstanceCount':  1,
            'InitialVariantWeight':  1,
        }],
    )

    sm_client.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG,
    )
    print(f'Deploying endpoint: {ENDPOINT_NAME}  (instance: {instance_type})')
    print('This takes ~5–8 minutes...')

    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=ENDPOINT_NAME, WaiterConfig={'Delay': 30, 'MaxAttempts': 30})
    print(f'Endpoint InService: {ENDPOINT_NAME}')

    if demo_mode:
        print('demo_mode=True — deleting endpoint to avoid charges...')
        time.sleep(10)
        teardown(sm_client)

    return ENDPOINT_NAME


def teardown(sm_client=None) -> None:
    """
    Delete endpoint, endpoint config, and model to stop billing.

    IMPORTANT: Run this when the endpoint is no longer needed.
    An idle ml.g4dn.xlarge costs ~$0.74/hr even with zero traffic.
    """
    if sm_client is None:
        sm_client = boto3.client('sagemaker')
    for fn, name in [
        (sm_client.delete_endpoint,        ENDPOINT_NAME),
        (sm_client.delete_endpoint_config, ENDPOINT_CONFIG),
        (sm_client.delete_model,           MODEL_NAME),
    ]:
        try:
            fn(**{list(fn.__code__.co_varnames)[1]: name})
            print(f'Deleted: {name}')
        except Exception as e:
            print(f'Could not delete {name}: {e}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket',        required=True)
    parser.add_argument('--region',        default='us-east-1')
    parser.add_argument('--instance-type', default='ml.g4dn.xlarge',
                        choices=['ml.g4dn.xlarge', 'ml.c5.large'])
    parser.add_argument('--demo',          action='store_true',
                        help='Deploy then immediately teardown (cost-free demo)')
    parser.add_argument('--teardown',      action='store_true',
                        help='Delete existing endpoint and resources')
    args = parser.parse_args()

    if args.teardown:
        teardown()
    else:
        deploy(args.bucket, args.region, args.instance_type, demo_mode=args.demo)
