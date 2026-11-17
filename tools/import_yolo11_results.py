"""Import the trained YOLO11s Kaggle ZIP without retraining or bulk extraction."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, check_class_names
from src.utils import metrics_table, save_json


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def unique_entry(names, suffix):
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {suffix}, found {matches}.')
    return matches[0]


def import_results(archive_path, project_root=PROJECT_ROOT, notebook_path=None):
    archive_path, project_root = Path(archive_path), Path(project_root).resolve()
    report = project_root / 'reports' / 'yolo11s'
    weights_dir = project_root / 'models' / 'yolo11s'
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        summary_entry = unique_entry(names, '/summary.json')
        validation_prefix = summary_entry.rsplit('/', 1)[0]
        best_entry = unique_entry(names, '/weights/best.pt')
        train_prefix = best_entry.rsplit('/weights/', 1)[0]
        args_entry = train_prefix + '/args.yaml'
        args = yaml.safe_load(archive.read(args_entry))
        if Path(args['model']).name != 'yolo11s.pt':
            raise ValueError(f'This importer expects YOLO11s, got {args["model"]}.')
        dataset = yaml.safe_load(archive.read('ppe_train.yaml'))
        check_class_names(dataset['names'])
        original = json.loads(archive.read(summary_entry))
        rows = list(csv.DictReader(io.StringIO(archive.read(train_prefix + '/results.csv').decode('utf-8-sig'))))
        best_epoch = int(max(rows, key=lambda row: float(row['metrics/mAP50-95(B)']))['epoch'])
        weights = archive.read(best_entry)
        weights_hash = hashlib.sha256(weights).hexdigest()
        weights_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = weights_dir / 'best.pt'
        if checkpoint.exists() and sha256_file(checkpoint) != weights_hash:
            raise FileExistsError(f'A different checkpoint already exists at {checkpoint}. Move it first.')
        checkpoint.write_bytes(weights)
        report.mkdir(parents=True, exist_ok=True)
        # Select known members and construct destinations ourselves; no extractall.
        for filename in ('args.yaml', 'results.csv', 'results.png'):
            (report / filename).write_bytes(archive.read(train_prefix + '/' + filename))
        for filename in ('confusion_matrix.png', 'confusion_matrix_normalized.png',
                         'BoxF1_curve.png', 'BoxPR_curve.png', 'per_class_map50.csv'):
            (report / filename).write_bytes(archive.read(validation_prefix + '/' + filename))
        (report / 'environment.txt').write_bytes(archive.read('environment.txt'))
        (report / 'source_summary.json').write_bytes(archive.read(summary_entry))
        (report / 'source_metrics_summary.csv').write_bytes(archive.read(validation_prefix + '/metrics_summary.csv'))
        (report / 'source_data.yaml').write_bytes(archive.read('ppe_train.yaml'))
        prediction_names = sorted(name for name in names
                                  if '_predictions/' in name and name.lower().endswith(('.jpg', '.png')))[:4]
        for index, name in enumerate(prediction_names, 1):
            output = report / 'examples' / f'prediction_{index:02d}{Path(name).suffix.lower()}'
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read(name))
    summary = dict(original)
    summary.update(model='yolo11s.pt', split='val', class_names=list(CLASS_NAMES),
                   epochs_completed=len(rows), best_epoch=best_epoch,
                   checkpoint_sha256=weights_hash, checkpoint_bytes=len(weights), test_evaluated=False)
    save_json(report / 'summary.json', summary)
    metrics_table(summary).to_csv(report / 'metrics_summary.csv', index=False)
    metadata = {
        'architecture': 'YOLO11s', 'task': 'detect', 'names': dict(enumerate(CLASS_NAMES)),
        'checkpoint': 'models/yolo11s/best.pt', 'sha256': weights_hash, 'bytes': len(weights),
        'training_model': args['model'], 'imgsz': args['imgsz'],
        'ultralytics': original['ultralytics'], 'torch': original['torch'],
        'archive_name': archive_path.name, 'archive_sha256': sha256_file(archive_path),
        'archive_checkpoint_entry': best_entry, 'split_reported': 'val',
        'corrections': [
            'Source summary.model and CSV Model said YOLO11n; corrected to YOLO11s using train args and checkpoint.',
            'Source validation/prediction directory names retain yolo11n; those are historical run names.',
        ],
    }
    if notebook_path:
        notebook_path = Path(notebook_path)
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        note = ('\n\nRepository import note: this run trained YOLO11s with four classes '
                '(boots, helmet, person, vest). Source labels have been corrected from YOLO11n to YOLO11s. '
                'Saved execution outputs are preserved as historical evidence and may still show the old label. '
                'The reported metrics are validation results; see reports/yolo11s/README.md.\n')
        for cell in notebook['cells']:
            source = ''.join(cell['source'])
            cell['source'] = source.replace('yolo11n', 'yolo11s').replace('YOLO11n', 'YOLO11s').splitlines(keepends=True)
        notebook['cells'][0]['source'].append(note)
        target = project_root / 'notebooks' / '03_training_yolo11_kaggle.ipynb'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        metadata['source_notebook_sha256'] = sha256_file(notebook_path)
        metadata['repository_notebook_sha256'] = sha256_file(target)
    save_json(weights_dir / 'metadata.json', metadata)
    return summary, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--notebook', type=Path)
    args = parser.parse_args(argv)
    summary, metadata = import_results(args.archive, notebook_path=args.notebook)
    print(metrics_table(summary).round(4).to_string(index=False))
    print(f'Checkpoint SHA256: {metadata["sha256"]}')
    print(f'Weights: {PROJECT_ROOT / metadata["checkpoint"]}')
    print(f'Reports: {PROJECT_ROOT / "reports" / "yolo11s"}')


if __name__ == '__main__':
    main()
