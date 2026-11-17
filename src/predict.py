"""Stream predictions for an image, directory, video or webcam."""
import argparse
from pathlib import Path

from src.config import DEFAULT_WEIGHTS, IMAGE_SIZE, PROJECT_ROOT
from src.utils import load_detector


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Image, directory, video, URL, or webcam index such as 0.')
    parser.add_argument('--weights', default=str(DEFAULT_WEIGHTS))
    parser.add_argument('--conf', type=float, default=.5)
    parser.add_argument('--iou', type=float, default=.7, help='NMS IoU threshold.')
    parser.add_argument('--imgsz', type=int, default=IMAGE_SIZE)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'runs' / 'predict' / 'yolo11s')
    parser.add_argument('--show', action='store_true', help='Display video/webcam frames in a local window.')
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error('--conf and --iou must be between 0 and 1.')
    model = load_detector(args.weights)
    source = int(args.source) if args.source.isdecimal() else args.source
    count, output = 0, None
    for result in model.predict(source=source, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                device=args.device, half=False, stream=True, save=True,
                                save_txt=True, save_conf=True, show=args.show,
                                project=str(args.output.parent), name=args.output.name):
        count += 1
        output = result.save_dir
    if count == 0:
        raise RuntimeError('No images or video frames were processed.')
    print(f'Processed {count} image(s)/frame(s). Outputs: {output}')


if __name__ == '__main__':
    main()
