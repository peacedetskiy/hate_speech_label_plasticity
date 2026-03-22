import argparse
import glob
import os
import sys

from pipeline import main


def natural_key(path: str):
    """
    Sort files like subset_1, subset_2, subset_10 correctly.
    """
    name = os.path.basename(path)
    parts = []
    current = ""
    is_digit = False

    for ch in name:
        if ch.isdigit():
            if not is_digit and current:
                parts.append(current)
                current = ch
            else:
                current += ch
            is_digit = True
        else:
            if is_digit and current:
                parts.append(int(current))
                current = ch
            else:
                current += ch
            is_digit = False

    if current:
        parts.append(int(current) if is_digit else current)

    return parts


def main_cli():
    parser = argparse.ArgumentParser(
        description="Run a single model across all experiment subsets"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name (e.g., llama3.2:3b)",
    )

    parser.add_argument(
        "--pattern",
        default="datasets/mhs_experiment_subset_*.parquet",
        help="File pattern for subsets",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )

    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern), key=natural_key)

    if not files:
        sys.exit(f"No dataset files found matching pattern: {args.pattern}")

    print("\n==============================")
    print(f"Model: {args.model}")
    print("Datasets to process:")
    for f in files:
        print(f" - {f}")
    print("==============================\n")

    for dataset_path in files:
        print(f"\n>>> Running {args.model} on {dataset_path}\n")
        main(
            model=args.model,
            dataset_path=dataset_path,
            overwrite=args.overwrite,
        )

    print("\n Finished running model on all subsets.")


if __name__ == "__main__":
    main_cli()