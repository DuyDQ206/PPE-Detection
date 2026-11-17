"""Defaults for the four-class YOLO11s PPE experiment."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / 'configs' / 'ppe4.yaml'
DEFAULT_WEIGHTS = PROJECT_ROOT / 'models' / 'yolo11s' / 'best.pt'
CLASS_NAMES = ('boots', 'helmet', 'person', 'vest')
IMAGE_SIZE = 640
BATCH_SIZE = 16
SEED = 42


def check_class_names(names):
    if isinstance(names, dict):
        indexed = {int(key): str(value) for key, value in names.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError('Class IDs must be contiguous from zero.')
        names = [indexed[i] for i in range(len(indexed))]
    if list(names) != list(CLASS_NAMES):
        raise ValueError(f'This checkpoint requires class IDs {dict(enumerate(CLASS_NAMES))}; got {names}.')
    return list(CLASS_NAMES)
