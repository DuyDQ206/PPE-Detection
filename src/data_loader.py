"""Validate the four-class dataset and relocate YAML paths for local execution."""
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

from src.config import PROJECT_ROOT, check_class_names


def prepare_dataset(yaml_path, required_splits, data_root=None):
    yaml_path = Path(yaml_path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f'Dataset YAML does not exist: {yaml_path}')
    config = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or 'names' not in config:
        raise ValueError('Dataset YAML must contain a names mapping.')
    if not all(key in config for key in ('train', 'val')):
        raise ValueError('Ultralytics requires both train and val keys in the dataset YAML, including for evaluation.')
    names = check_class_names(config['names'])
    if int(config.get('nc', len(names))) != len(names):
        raise ValueError('nc does not match names.')
    root = Path(data_root) if data_root else Path(config.get('path', '.'))
    if not root.is_absolute():
        root = (Path.cwd() if data_root else yaml_path.parent) / root
    root = root.resolve()
    resolved = {'path': str(root), 'nc': len(names), 'names': names}
    for split in ('train', 'val', 'test'):
        value = config.get(split)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f'{split}: expected an image directory path.')
        folder = Path(value)
        declared_absolute = PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
        if data_root and declared_absolute:
            parts = value.replace('\\', '/').split('/')
            marker = next((part for part in parts if part in ('train', 'valid', 'val', 'test')), None)
            if marker is None:
                raise ValueError(f'Cannot relocate {split}: {value}')
            folder = Path(*parts[parts.index(marker):])
        folder = (folder if folder.is_absolute() else root / folder).resolve()
        resolved[split] = str(folder)
    for split in required_splits:
        if split not in resolved or not Path(resolved[split]).is_dir():
            raise FileNotFoundError(f'Missing {split} images under {root}. Supply --data and/or --data-root.')
        if not any(p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
                   for p in Path(resolved[split]).rglob('*') if p.is_file()):
            raise ValueError(f'No images found for {split}: {resolved[split]}')
    if 'train' in required_splits and resolved['train'] == resolved.get('val'):
        raise ValueError('Training and validation must use different directories.')
    # Do not carry over download hooks from an external dataset YAML.
    text = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    output = PROJECT_ROOT / 'runs' / 'datasets' / (hashlib.sha256(text.encode()).hexdigest()[:16] + '.yaml')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    return output
