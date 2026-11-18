"""Model loading and evaluation exports shared by the command-line tools."""
import json
from pathlib import Path

import pandas as pd

from src.config import check_class_names


def load_detector(weights):
    from ultralytics import YOLO

    weights = Path(weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(
            f'Checkpoint missing: {weights}. Copy ppe_yolo11s-2/weights/best.pt from the Kaggle ZIP '
            'to models/yolo11s/best.pt, or supply --weights with your checkpoint path.'
        )
    model = YOLO(str(weights))
    check_class_names(model.names)
    return model


def collect_metrics(metrics, network, image_size):
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    gflops = float(get_flops(network, imgsz=image_size))
    if not gflops > 0:
        raise RuntimeError('GFLOPs profiling failed. Check the ultralytics-thop installation.')
    params = int(get_num_params(network))
    speed = {key: float(metrics.speed[key]) for key in ('preprocess', 'inference', 'postprocess')}
    return {
        'params': params, 'params_millions': params / 1e6, 'gflops': gflops,
        'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
        'f1': float(sum(metrics.box.f1) / len(metrics.box.f1)) if len(metrics.box.f1) else 0.0,
        'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
        'latency_inference_ms': speed['inference'],
        'latency_preprocess_ms': speed['preprocess'],
        'latency_postprocess_ms': speed['postprocess'],
        'latency_pipeline_ms': sum(speed.values()),
    }


def metrics_table(summary):
    return pd.DataFrame([{
        'Model': 'YOLO11s', 'Params (M)': summary['params_millions'], 'GFLOPs': summary['gflops'],
        'P': summary['precision'], 'R': summary['recall'], 'F1': summary['f1'],
        'mAP50': summary['mAP50'], 'mAP50-95': summary['mAP50_95'],
        'Latency (ms/img)': summary['latency_inference_ms'],
    }])


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
