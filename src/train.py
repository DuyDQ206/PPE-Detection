"""Train YOLO11s locally using the same baseline settings as the Kaggle run."""
import argparse
from pathlib import Path

from src.config import BATCH_SIZE, DEFAULT_DATA, IMAGE_SIZE, PROJECT_ROOT, SEED
from src.data_loader import prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(DEFAULT_DATA))
    parser.add_argument('--data-root', help='Override the dataset root when relocating a Kaggle dataset.')
    parser.add_argument('--model', default='yolo11s.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--imgsz', type=int, default=IMAGE_SIZE)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', default=None, help='For example: cpu or 0. Default: automatic.')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--project', default=str(PROJECT_ROOT / 'runs' / 'train'))
    parser.add_argument('--name', default='ppe_yolo11s')
    args = parser.parse_args(argv)
    args.project = str(Path(args.project).expanduser().resolve())
    data = prepare_dataset(args.data, ('train', 'val'), args.data_root)
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, project=args.project, name=args.name,
                pretrained=True, patience=args.patience, save=True, plots=True, seed=SEED)
    print(f'Best checkpoint: {model.trainer.best}')


if __name__ == '__main__':
    main()
