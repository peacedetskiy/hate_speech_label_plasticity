# results_utils.py

from __future__ import annotations

import re
import pandas as pd


ALL_CONDITIONS = (
    "dataset_random",
    "pool_random_0",
    "pool_random_1",
    "neutral",
    "original",
    "inverted",
)

_ALLOWED_OUTPUT_VALUES = {"Hate Speech", "Not Hate Speech", ""}


def is_result_column(series: pd.Series) -> bool:
    values = {
        str(v).strip()
        for v in series.dropna().unique().tolist()
    }
    return bool(values) and values.issubset(_ALLOWED_OUTPUT_VALUES)


def get_result_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if not any(col.endswith(f"_{cond}") for cond in ALL_CONDITIONS):
            continue
        if is_result_column(df[col]):
            cols.append(col)
    return cols


def split_result_column(col: str) -> tuple[str, str]:
    for cond in sorted(ALL_CONDITIONS, key=len, reverse=True):
        suffix = f"_{cond}"
        if col.endswith(suffix):
            return col[: -len(suffix)], cond
    raise ValueError(f"Not a recognized result column: {col}")


def get_model_names(df: pd.DataFrame) -> list[str]:
    models = {split_result_column(col)[0] for col in get_result_columns(df)}
    return sorted(models)


def get_columns_for_model(df: pd.DataFrame, model: str) -> dict[str, str]:
    out = {}
    for col in get_result_columns(df):
        m, cond = split_result_column(col)
        if m == model:
            out[cond] = col
    return out


def model_condition_crosstabs(df: pd.DataFrame, label_col: str = "human_label") -> dict:
    out = {}
    for model in get_model_names(df):
        out[model] = {}
        for cond, col in get_columns_for_model(df, model).items():
            out[model][cond] = pd.crosstab(df[label_col], df[col], dropna=False)
    return out