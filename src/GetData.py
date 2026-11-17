"""Download the raw Roboflow dataset; preprocessing is still required."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if not api_key:
        raise RuntimeError('Set ROBOFLOW_API_KEY in your environment or .env file.')
    from roboflow import Roboflow
    version = Roboflow(api_key=api_key).workspace('constructionsite').project('ppe-fruwx').version(5)
    version.download(target_format='yolov8', location=str(args.output))


if __name__ == '__main__':
    main()
