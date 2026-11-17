"""Convenience entry point: python main.py predict/train/evaluate [options]."""
import argparse
import importlib


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('predict', 'train', 'evaluate'))
    parser.add_argument('options', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    importlib.import_module(f'src.{args.command}').main(args.options)


if __name__ == '__main__':
    main()
