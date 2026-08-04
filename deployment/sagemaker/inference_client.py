"""
Nudge CATE inference client.

Preprocesses raw customer features, calls the SageMaker/Triton endpoint,
and returns a structured prediction with persuadability label and action.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Literal

import boto3

ENDPOINT_NAME = "nudge-cate-endpoint"

FEATURE_ORDER = [
    'recency', 'history', 'mens', 'womens', 'newbie',
    'recency_bucket', 'spend_tier',
    'zip_urban', 'zip_suburban', 'zip_rural',
    'ch_phone', 'ch_web', 'ch_multi',
    'newbie_x_spend_tier', 'recency_x_spend_tier',
]

# CATE thresholds — tuned on Hillstrom test set
PERSUADABLE_THRESHOLD  =  0.01   # CATE > this → send promotion
SLEEPING_DOG_THRESHOLD = -0.01   # CATE < this → actively withhold

PersuadabilityLabel = Literal['persuadable', 'neutral', 'sleeping_dog']


@dataclass
class CATEPrediction:
    cate_score:            float
    persuadability:        PersuadabilityLabel
    send_promotion:        bool
    action_recommendation: str
    confidence:            str   # 'high' | 'medium' | 'low' based on score magnitude


def _derive_features(raw: dict) -> dict:
    """
    Derive engineered features from raw customer record.
    Mirrors queries/02_hillstrom_features.sql — keep in sync.
    """
    history = float(raw.get('history', 0))
    recency = int(raw.get('recency', 0))
    newbie  = int(raw.get('newbie', 0))

    history_segment = raw.get('history_segment', '$0 - $100')
    spend_tier_map = {
        '$0 - $100': 0, '$100 - $200': 1, '$200 - $350': 2,
        '$350 - $500': 3, '$500 - $750': 4, '$750 - $1,000': 5, '$1,000 +': 6,
    }
    spend_tier = spend_tier_map.get(history_segment, 0)

    zip_code = raw.get('zip_code', '')
    channel  = raw.get('channel', '')

    return {
        'recency':               recency,
        'history':               history,
        'mens':                  int(raw.get('mens', 0)),
        'womens':                int(raw.get('womens', 0)),
        'newbie':                newbie,
        'recency_bucket':        0 if recency <= 3 else 1 if recency <= 6 else 2 if recency <= 12 else 3,
        'spend_tier':            spend_tier,
        'zip_urban':             int(zip_code == 'Urban'),
        # The Hillstrom source data misspells this as 'Surburban' and training
        # encodes that literal, so accept both spellings from API callers.
        'zip_suburban':          int(zip_code in ('Surburban', 'Suburban')),
        'zip_rural':             int(zip_code == 'Rural'),
        'ch_phone':              int(channel == 'Phone'),
        'ch_web':                int(channel == 'Web'),
        'ch_multi':              int(channel == 'Multichannel'),
        'newbie_x_spend_tier':   newbie * spend_tier,
        'recency_x_spend_tier':  recency * spend_tier,
    }


def _label(cate: float) -> tuple[PersuadabilityLabel, bool, str, str]:
    if cate > PERSUADABLE_THRESHOLD:
        label     = 'persuadable'
        send      = True
        action    = f'Send promotion. Expected incremental conversion lift: +{cate:.3f}.'
        magnitude = abs(cate)
        conf      = 'high' if magnitude > 0.05 else 'medium' if magnitude > 0.02 else 'low'
    elif cate < SLEEPING_DOG_THRESHOLD:
        label  = 'sleeping_dog'
        send   = False
        action = 'Withhold promotion. Customer converts without incentive; discount erodes margin.'
        conf   = 'high' if abs(cate) > 0.02 else 'medium'
    else:
        label  = 'neutral'
        send   = False
        action = 'No action. Predicted lift is below cost-benefit threshold.'
        conf   = 'low'
    return label, send, action, conf


def predict_single(raw_customer: dict, endpoint_name: str = ENDPOINT_NAME) -> CATEPrediction:
    """
    Score a single customer and return a structured CATE prediction.

    Parameters
    ----------
    raw_customer : dict with keys matching Hillstrom schema
                   (recency, history, history_segment, mens, womens,
                    zip_code, newbie, channel)
    """
    features = _derive_features(raw_customer)
    feature_vector = np.array(
        [[features[col] for col in FEATURE_ORDER]], dtype=np.float32
    )

    payload = {
        "inputs": [{
            "name":     "input__0",
            "shape":    [1, len(FEATURE_ORDER)],
            "datatype": "FP32",
            "data":     feature_vector.flatten().tolist(),
        }]
    }

    runtime  = boto3.client('sagemaker-runtime')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/octet-stream',
        Body=json.dumps(payload),
    )
    result = json.loads(response['Body'].read())
    cate   = float(result['outputs'][0]['data'][0])

    label, send, action, conf = _label(cate)
    return CATEPrediction(
        cate_score=round(cate, 6),
        persuadability=label,
        send_promotion=send,
        action_recommendation=action,
        confidence=conf,
    )


def predict_batch(customers: list[dict], endpoint_name: str = ENDPOINT_NAME) -> list[CATEPrediction]:
    """Score a batch of customers (up to max_batch_size=1024)."""
    vectors = np.array(
        [[_derive_features(c)[col] for col in FEATURE_ORDER] for c in customers],
        dtype=np.float32,
    )
    payload = {
        "inputs": [{
            "name":     "input__0",
            "shape":    list(vectors.shape),
            "datatype": "FP32",
            "data":     vectors.flatten().tolist(),
        }]
    }
    runtime  = boto3.client('sagemaker-runtime')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/octet-stream',
        Body=json.dumps(payload),
    )
    result = json.loads(response['Body'].read())
    cates  = result['outputs'][0]['data']

    predictions = []
    for c in cates:
        label, send, action, conf = _label(c)
        predictions.append(CATEPrediction(
            cate_score=round(c, 6),
            persuadability=label,
            send_promotion=send,
            action_recommendation=action,
            confidence=conf,
        ))
    return predictions


if __name__ == '__main__':
    test_customer = {
        'recency': 6, 'history': 350.0, 'history_segment': '$350 - $500',
        'mens': 1, 'womens': 0, 'zip_code': 'Urban',
        'newbie': 0, 'channel': 'Web',
    }
    print('Test customer:', test_customer)
    pred = predict_single(test_customer)
    print(asdict(pred))
