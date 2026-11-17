"""Evaluate the trained PPE checkpoint and export all eight requested metrics."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import BATCH_SIZE, CLASS_NAMES, DEFAULT_DATA, DEFAULT_WEIGHTS, IMAGE_SIZE, PROJECT_ROOT
from src.data_loader import prepare_dataset
from src.utils import collect_metrics, load_detector, metrics_table, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default=str(DEFAULT_WEIGHTS))
    parser.add_argument('--data', default=str(DEFAULT_DATA))
    parser.add_argument('--data-root')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--imgsz', type=int, default=IMAGE_SIZE)
    parser.add_argument('--device', default=None)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'runs' / 'evaluate' / 'yolo11s')
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    data = prepare_dataset(args.data, (args.split,), args.data_root)
    model = load_detector(args.weights)
    model.fuse()
    metrics = model.val(data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
                        device=args.device, workers=args.workers, half=False, plots=True,
                        project=str(args.output.parent), name=args.output.name)
    summary = collect_metrics(metrics, model.model, args.imgsz)
    import torch
    import ultralytics

    device = next(model.model.parameters()).device
    summary.update(
        model='yolo11s.pt', weights=str(Path(args.weights).resolve()), split=args.split,
        class_names=list(CLASS_NAMES), imgsz=args.imgsz, batch=args.batch,
        params_model_state='fused', flops_input_shape=[1, 3, args.imgsz, args.imgsz],
        f1_definition='Mean per-class F1 at the Ultralytics max-mean-F1 operating point (IoU 0.5)',
        latency_method='Validation average per image; excludes file I/O; pipeline = preprocess + inference + postprocess',
        latency_batch_size=args.batch, latency_precision='FP32', device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
        ultralytics=ultralytics.__version__, torch=torch.__version__,
        source_data_yaml=str(Path(args.data).resolve()), resolved_data_yaml=str(data),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    output = Path(metrics.save_dir)
    save_json(output / 'summary.json', summary)
    table = metrics_table(summary)
    table.to_csv(output / 'metrics_summary.csv', index=False)
    evaluated = {int(cid): i for i, cid in enumerate(metrics.box.ap_class_index)}
    rows = []
    for cid, name in enumerate(CLASS_NAMES):
        i = evaluated.get(cid)
        rows.append({'class_id': cid, 'class_name': name,
                     'P': float(metrics.box.p[i]) if i is not None else None,
                     'R': float(metrics.box.r[i]) if i is not None else None,
                     'F1': float(metrics.box.f1[i]) if i is not None else None,
                     'mAP50': float(metrics.box.ap50[i]) if i is not None else None,
                     'mAP50-95': float(metrics.box.ap[i]) if i is not None else None})
    pd.DataFrame(rows).to_csv(output / 'per_class_metrics.csv', index=False)
    print(table.round(4).to_string(index=False))
    print(f'Outputs: {output}')


if __name__ == '__main__':
    main()
