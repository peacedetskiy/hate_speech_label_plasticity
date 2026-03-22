import argparse
import os
import random

import pandas as pd

from schema import normalize


def validate_columns(df: pd.DataFrame):
    required = ["text", "human_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def stratified_sample(df: pd.DataFrame, per_class: int, seed: int):
    rng = random.Random(seed)

    labels = sorted(df["human_label"].dropna().unique())

    if len(labels) < 3:
        print(f"Warning: expected 3 classes, found {labels}")

    subsets = {}

    for label in labels:
        subset = df[df["human_label"] == label]
        if len(subset) < per_class:
            raise ValueError(
                f"Not enough samples for label '{label}'. "
                f"Required={per_class}, available={len(subset)}"
            )
        subsets[label] = subset.sample(n=per_class, random_state=seed)

    return pd.concat(subsets.values()).sample(frac=1, random_state=seed)


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
                    f"Subset {i+1}: not enough unused samples for label '{label}'. "
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
    print(df["human_label"].value_counts())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="datasets")
    parser.add_argument("--prefix", default="experiment")
    parser.add_argument("--subsets", type=int, default=3)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
        filename = f"{args.prefix}_subset_{i}.parquet"
        path = os.path.join(args.output_dir, filename)

        subset.to_parquet(path)
        paths.append(path)

        print_stats(subset, filename)

    # Check overlap
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