"""
CloudWatch alarms and SageMaker Model Monitor setup for the Nudge CATE endpoint.

Monitors:
  - Endpoint latency (p99 > 100ms → alarm)
  - Invocation errors (> 1% error rate → alarm)
  - CATE score distribution drift (daily baseline comparison)
"""

import boto3
from datetime import datetime

ENDPOINT_NAME   = "nudge-cate-endpoint"
ALARM_SNS_ARN   = ""  # fill with your SNS topic ARN for alerts
MONITOR_BUCKET  = ""  # fill with your S3 bucket name


def create_latency_alarm(
    cw_client,
    endpoint_name: str = ENDPOINT_NAME,
    threshold_ms: float = 100.0,
    sns_arn: str = ALARM_SNS_ARN,
) -> None:
    """Alert if p99 ModelLatency exceeds threshold_ms."""
    cw_client.put_metric_alarm(
        AlarmName=f'{endpoint_name}-p99-latency',
        AlarmDescription=f'Nudge CATE endpoint p99 latency > {threshold_ms}ms',
        Namespace='AWS/SageMaker',
        MetricName='ModelLatency',
        Dimensions=[
            {'Name': 'EndpointName',  'Value': endpoint_name},
            {'Name': 'VariantName',   'Value': 'AllTraffic'},
        ],
        Statistic='p99',
        Period=60,
        EvaluationPeriods=3,
        Threshold=threshold_ms * 1000,  # CloudWatch uses microseconds
        ComparisonOperator='GreaterThanThreshold',
        AlarmActions=[sns_arn] if sns_arn else [],
        TreatMissingData='notBreaching',
    )
    print(f'Alarm created: {endpoint_name}-p99-latency  (threshold: {threshold_ms}ms)')


def create_error_rate_alarm(
    cw_client,
    endpoint_name: str = ENDPOINT_NAME,
    error_pct_threshold: float = 1.0,
    sns_arn: str = ALARM_SNS_ARN,
) -> None:
    """Alert if invocation error rate exceeds error_pct_threshold %."""
    cw_client.put_metric_alarm(
        AlarmName=f'{endpoint_name}-error-rate',
        AlarmDescription=f'Nudge CATE endpoint error rate > {error_pct_threshold}%',
        Namespace='AWS/SageMaker',
        MetricName='Invocation4XXErrors',
        Dimensions=[
            {'Name': 'EndpointName', 'Value': endpoint_name},
            {'Name': 'VariantName',  'Value': 'AllTraffic'},
        ],
        Statistic='Sum',
        Period=300,
        EvaluationPeriods=2,
        Threshold=error_pct_threshold,
        ComparisonOperator='GreaterThanThreshold',
        AlarmActions=[sns_arn] if sns_arn else [],
        TreatMissingData='notBreaching',
    )
    print(f'Alarm created: {endpoint_name}-error-rate')


def setup_data_capture(sm_client, bucket: str = MONITOR_BUCKET) -> None:
    """
    Enable SageMaker Data Capture to log inference inputs/outputs to S3.
    Required for Model Monitor drift detection.
    """
    if not bucket:
        print('MONITOR_BUCKET not set — skipping data capture setup.')
        return

    sm_client.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        RetainDeploymentConfig=True,
        DeploymentConfig={
            'DataCaptureConfig': {
                'EnableCapture': True,
                'InitialSamplingPercentage': 100,
                'DestinationS3Uri': f's3://{bucket}/nudge/data-capture/',
                'CaptureOptions': [
                    {'CaptureMode': 'Input'},
                    {'CaptureMode': 'Output'},
                ],
                'CaptureContentTypeHeader': {
                    'JsonContentTypes': ['application/json'],
                },
            }
        },
    )
    print(f'Data capture enabled → s3://{bucket}/nudge/data-capture/')


def create_drift_monitor(
    sm_client,
    bucket: str = MONITOR_BUCKET,
    baseline_uri: str = '',
) -> None:
    """
    Create a SageMaker Model Monitor schedule for daily CATE distribution checks.
    Requires a baseline statistics file (generate with ModelMonitor.suggest_baseline()).
    """
    if not bucket or not baseline_uri:
        print('MONITOR_BUCKET or baseline_uri not set — skipping drift monitor.')
        return

    monitor_name = f'{ENDPOINT_NAME}-drift-monitor'
    sm_client.create_monitoring_schedule(
        MonitoringScheduleName=monitor_name,
        MonitoringScheduleConfig={
            'ScheduleConfig': {'ScheduleExpression': 'cron(0 8 * * ? *)'},  # daily 8am UTC
            'MonitoringJobDefinition': {
                'MonitoringInputs': [{
                    'EndpointInput': {
                        'EndpointName':             ENDPOINT_NAME,
                        'LocalPath':                '/opt/ml/processing/input/endpoint',
                        'S3InputMode':              'File',
                        'S3DataDistributionType':   'FullyReplicated',
                    }
                }],
                'MonitoringOutputConfig': {
                    'MonitoringOutputs': [{
                        'S3Output': {
                            'S3Uri':        f's3://{bucket}/nudge/monitor-results/',
                            'LocalPath':    '/opt/ml/processing/output',
                            'S3UploadMode': 'EndOfJob',
                        }
                    }]
                },
                'MonitoringResources': {
                    'ClusterConfig': {
                        'InstanceCount':  1,
                        'InstanceType':   'ml.m5.xlarge',
                        'VolumeSizeInGB': 20,
                    }
                },
                'MonitoringAppSpecification': {
                    'ImageUri': '156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer',
                },
                'BaselineConfig': {
                    'StatisticsResource': {'S3Uri': f'{baseline_uri}/statistics.json'},
                    'ConstraintsResource': {'S3Uri': f'{baseline_uri}/constraints.json'},
                },
                'RoleArn': '',  # fill with execution role ARN
            }
        }
    )
    print(f'Drift monitor scheduled: {monitor_name} (daily 08:00 UTC)')


if __name__ == '__main__':
    cw = boto3.client('cloudwatch')
    sm = boto3.client('sagemaker')
    create_latency_alarm(cw)
    create_error_rate_alarm(cw)
    setup_data_capture(sm)
    print('\nMonitoring setup complete.')
    print('Next: set ALARM_SNS_ARN and MONITOR_BUCKET, then re-run for full config.')
