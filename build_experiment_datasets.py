# build_experiment_datasets.py

import argparse
import os
from pathlib import Path

import pandas as pd

from schema import normalize, detectDataset


def validate_columns(df: pd.DataFrame):
    required = ["text", "human_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    na_count = int(df["human_label"].isna().sum())
    if na_count:
        raise ValueError(
            f"'human_label' contains {na_count} missing values after normalize(). "
            "Refusing to build subsets from invalid labels."
        )


def build_subsets(
    df: pd.DataFrame,
    num_subsets: int,
    per_class: int,
    base_seed: int,
):
    used_indices = set()
    subsets = []

    labels = sorted(df["human_label"].dropna().unique())

    for i in range(num_subsets):
        seed = base_seed + i
        parts = []

        for label in labels:
            pool = df[
                (df["human_label"] == label)
                & (~df.index.isin(used_indices))
            ]

            if len(pool) < per_class:
                raise ValueError(
                    f"Subset {i + 1}: not enough unused samples for label '{label}'. "
                    f"Needed={per_class}, remaining={len(pool)}"
                )

            sampled = pool.sample(n=per_class, random_state=seed)
            parts.append(sampled)
            used_indices.update(sampled.index)

        subset_df = pd.concat(parts).sample(frac=1, random_state=seed)
        subsets.append(subset_df)

    return subsets


def print_stats(df: pd.DataFrame, name: str):
    print(f"\n{name}")
    print(f"Total rows: {len(df)}")
    print("Class distribution:")
    print(df["human_label"].value_counts(dropna=False).sort_index())


def default_prefix_for_input(input_path: str) -> str:
    dataset_key = detectDataset(input_path)
    if dataset_key == "popquorn":
        return "popquorn_experiment_v2"
    if dataset_key == "measuring_hate_speech":
        return "mhs_experiment_v2"
    return f"{Path(input_path).stem}_experiment_v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="datasets")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--subsets", type=int, default=3)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing subset files",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prefix = args.prefix or default_prefix_for_input(args.input)

    print("Loading dataset...")
    df = pd.read_parquet(args.input)
    df = normalize(df, args.input)

    validate_columns(df)

    print("Building subsets...")
    subsets = build_subsets(
        df,
        num_subsets=args.subsets,
        per_class=args.per_class,
        base_seed=args.seed,
    )

    paths = []

    for i, subset in enumerate(subsets, start=1):
        filename = f"{prefix}_subset_{i}.parquet"
        path = os.path.join(args.output_dir, filename)

        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}\n"
                "Use a different --prefix for testing, or pass --overwrite explicitly."
            )

        subset.to_parquet(path)
        paths.append(path)
        print_stats(subset, filename)

    all_indices = [set(s.index) for s in subsets]
    overlap = set.intersection(*all_indices) if len(all_indices) > 1 else set()

    print("\nOverlap check:")
    if overlap:
        print(f"WARNING: {len(overlap)} overlapping rows found")
    else:
        print("OK: no overlap between subsets")

    print("\nSaved files:")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()