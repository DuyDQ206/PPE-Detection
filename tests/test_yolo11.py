"""Offline regression checks; no checkpoint download or model training required."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
import pandas as pd
import yaml

from src.config import CLASS_NAMES, PROJECT_ROOT, check_class_names
from src.data_loader import prepare_dataset
from src.utils import collect_metrics
from tools.import_yolo11_results import import_results


class Yolo11Checks(unittest.TestCase):
    def test_class_order_and_eight_class_dataset_are_rejected(self):
        self.assertEqual(check_class_names(dict(enumerate(CLASS_NAMES))), list(CLASS_NAMES))
        for names in (list(reversed(CLASS_NAMES)), ['boots', 'helmet', 'helmet on', 'no boots',
                                                     'no helmet', 'no vest', 'person', 'vest']):
            with self.assertRaises(ValueError):
                check_class_names(names)

    def test_relocating_kaggle_paths_does_not_change_classes_or_run_download_hooks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ('train', 'valid'):
                images = root / split / 'images'
                images.mkdir(parents=True)
                Image.new('RGB', (8, 8)).save(images / 'sample.jpg')
            config = root / 'data.yaml'
            config.write_text(yaml.safe_dump({
                'path': '/kaggle/input/old', 'train': '/kaggle/input/old/train/images',
                'val': '/kaggle/input/old/valid/images', 'names': list(CLASS_NAMES),
                'download': 'raise RuntimeError("must never execute")',
            }), encoding='utf-8')
            with patch('src.data_loader.PROJECT_ROOT', root):
                result = yaml.safe_load(prepare_dataset(config, ('train', 'val'), root).read_text())
            self.assertEqual(result['val'], str((root / 'valid/images').resolve()))
            self.assertEqual(result['names'], list(CLASS_NAMES))
            self.assertNotIn('download', result)
            with self.assertRaises(FileNotFoundError):
                prepare_dataset(config, ('test',), root)

    def test_macro_f1_uses_per_class_values_and_pipeline_excludes_loss(self):
        metric = SimpleNamespace(
            box=SimpleNamespace(mp=.625, mr=.625, f1=[.4, .4], map50=.82, map=.49),
            speed={'preprocess': 1., 'inference': 9., 'postprocess': 2., 'loss': 99.},
        )
        profiler = SimpleNamespace(get_flops=lambda model, imgsz: 21.4,
                                   get_num_params=lambda model: 9_414_348)
        with patch.dict(sys.modules, {'ultralytics.utils.torch_utils': profiler}):
            summary = collect_metrics(metric, object(), 640)
            self.assertAlmostEqual(summary['f1'], .4)
            self.assertAlmostEqual(summary['latency_pipeline_ms'], 12.)
            self.assertAlmostEqual(summary['latency_inference_ms'], 9.)
            profiler.get_flops = lambda model, imgsz: 0.
            with self.assertRaises(RuntimeError):
                collect_metrics(metric, object(), 640)

    def test_import_keeps_original_evidence_and_corrects_model_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'results.zip'
            original = json.loads((PROJECT_ROOT / 'reports/yolo11s/source_summary.json').read_text())
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('ppe/weights/best.pt', b'checkpoint-fixture')
                z.writestr('ppe/args.yaml', 'model: yolo11s.pt\nimgsz: 640\n')
                z.writestr('ppe_train.yaml', yaml.safe_dump({'names': list(CLASS_NAMES)}))
                z.writestr('ppe/results.csv', 'epoch,metrics/mAP50-95(B)\n1,0.3\n2,0.5\n')
                z.writestr('ppe/results.png', b'plot-fixture')
                z.writestr('val/summary.json', json.dumps(original))
                for name in ('confusion_matrix.png', 'confusion_matrix_normalized.png',
                             'BoxF1_curve.png', 'BoxPR_curve.png', 'per_class_map50.csv'):
                    z.writestr('val/' + name, b'fixture')
                z.writestr('val/metrics_summary.csv', 'Model\nYOLO11n\n')
                z.writestr('environment.txt', 'ultralytics==8.4.215\n')
                z.writestr('../../must-not-extract.txt', b'ignored')
            summary, metadata = import_results(archive, root)
            self.assertEqual(summary['model'], 'yolo11s.pt')
            self.assertEqual(summary['best_epoch'], 2)
            self.assertFalse(summary['test_evaluated'])
            self.assertEqual(summary['precision'], original['precision'])
            saved_source = json.loads((root / 'reports/yolo11s/source_summary.json').read_text())
            self.assertEqual(saved_source, original)
            self.assertEqual(metadata['sha256'], hashlib.sha256(b'checkpoint-fixture').hexdigest())
            self.assertEqual(pd.read_csv(root / 'reports/yolo11s/metrics_summary.csv').iloc[0]['Model'], 'YOLO11s')
            checkpoint = root / 'models/yolo11s/best.pt'
            checkpoint.write_bytes(b'a-different-model')
            with self.assertRaises(FileExistsError):
                import_results(archive, root)

    def test_saved_report_matches_original_measurements(self):
        folder = PROJECT_ROOT / 'reports/yolo11s'
        summary = json.loads((folder / 'summary.json').read_text())
        original = json.loads((folder / 'source_summary.json').read_text())
        for key in ('params', 'gflops', 'precision', 'recall', 'f1', 'mAP50', 'mAP50_95', 'latency_inference_ms'):
            self.assertEqual(summary[key], original[key])
        table = pd.read_csv(folder / 'metrics_summary.csv')
        self.assertAlmostEqual(table.iloc[0]['F1'], summary['f1'])
        self.assertEqual(summary['split'], 'val')
        self.assertEqual(summary['class_names'], list(CLASS_NAMES))

    def test_predict_resolves_output_and_converts_webcam_index(self):
        from src.predict import main
        model = SimpleNamespace()
        calls = {}

        def predict(**kwargs):
            calls.update(kwargs)
            yield SimpleNamespace(save_dir=str(Path(kwargs['project']) / kwargs['name']))

        model.predict = predict
        with patch('src.predict.load_detector', return_value=model):
            main(['--source', '0', '--output', 'runs/predict/unit-check'])
        self.assertEqual(calls['source'], 0)
        self.assertTrue(Path(calls['project']).is_absolute())
        self.assertEqual(Path(calls['project']) / calls['name'], (Path.cwd() / 'runs/predict/unit-check').resolve())

    def test_notebook_code_compiles_and_cli_help_needs_no_model(self):
        notebook = json.loads((PROJECT_ROOT / 'notebooks/03_training_yolo11_kaggle.ipynb').read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        for command in ('train', 'evaluate', 'predict'):
            result = subprocess.run([sys.executable, 'main.py', command, '--help'],
                                    cwd=PROJECT_ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
