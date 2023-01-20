import argparse
import pandas as pd
from pathlib import Path
from clone_classifier import CloneClassifier


ROOT = Path(__file__).parent


def main():
    """Run the clone classifier on specified input file.

    By default, the input file is the C4 dataset located in examples/c4.csv,
    and the output file is located in results/c4.csv.

    Args:
        max_token_size: the maximum token size for the input data (default: 512)
        fp16: a flag to enable half-precision (default: False)
        input: the path to the input CSV file (default: 'examples/c4.csv')
        output_dir: the directory path to save the output file (default: 'results/')
        per_device_eval_batch_size: the batch size for evaluation per device (default: 32)
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--max_token_size", type=int, metavar="", default=512, help="Max token size"
    )
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision")
    parser.add_argument(
        "--input",
        type=str,
        metavar="",
        default=f"{ROOT / 'examples/c4.csv'}",
        help="Input file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        metavar="",
        default=f"{ROOT / 'results/'}",
        help="Output directory path",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        metavar="",
        default=32,
        help="Batch size per device for evaluation",
    )

    args = parser.parse_args()
    df = pd.read_csv(args.input)

    clone_classifier = CloneClassifier(args)
    clone_classifier.predict(df, save_filename="results.csv")

    return


if __name__ == "__main__":
    main()
