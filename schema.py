# schema.py

import logging
import pandas as pd


CANONICAL = {
    "text": "text",
    "label": "human_label",
    "age": "annotator_age",
    "gender": "annotator_gender",
    "race": "annotator_race",
    "ann_id": "annotator_id",
}

DATASET_SCHEMAS = {
    "measuring_hate_speech": {
        "title": CANONICAL["text"],
        "hatespeech": CANONICAL["label"],
        "annotator_age": CANONICAL["age"],
        "annotator_gender": CANONICAL["gender"],
        "annotator_id": CANONICAL["ann_id"],
    },
    "popquorn": {
        "title": CANONICAL["text"],
        "offensiveness": CANONICAL["label"],
        "age": CANONICAL["age"],
        "gender": CANONICAL["gender"],
        "race": CANONICAL["race"],
        "user_id": CANONICAL["ann_id"],
    },
}


def detectDataset(filename: str) -> str | None:
    base = filename.lower().replace("-", "_").replace(" ", "_")
    for key in DATASET_SCHEMAS:
        if key in base:
            return key
    return None


def _reconstruct_mhs_race(df: pd.DataFrame) -> pd.DataFrame:
    race_map = {
        "annotator_race_asian": "Asian",
        "annotator_race_black": "Black or African American",
        "annotator_race_latinx": "Hispanic or Latino",
        "annotator_race_middle_eastern": "Middle Eastern",
        "annotator_race_native_american": "Native American",
        "annotator_race_pacific_islander": "Mixed / Other",
        "annotator_race_white": "White",
        "annotator_race_other": "Mixed / Other",
    }

    if "annotator_race" in df.columns:
        return df

    race_cols = [col for col in race_map if col in df.columns]
    if not race_cols:
        return df

    def pick_race(row):
        for col in race_cols:
            val = row[col]
            if val is True or val == 1:
                return race_map[col]
        return "unknown"

    df = df.copy()
    df["annotator_race"] = df[race_cols].apply(pick_race, axis=1)
    return df


def _normalize_popquorn_race(df: pd.DataFrame) -> pd.DataFrame:
    if "annotator_race" not in df.columns:
        return df

    race_map = {
        "Arab American": "Middle Eastern",
        "Middle Eastern / North African": "Middle Eastern",
        "Latino": "Hispanic or Latino",
        "Latinx": "Hispanic or Latino",
        "Other": "Mixed / Other",
        "Mixed": "Mixed / Other",
    }

    df = df.copy()
    df["annotator_race"] = (
        df["annotator_race"]
        .astype(str)
        .str.strip()
        .replace(race_map)
    )
    return df


def _map_popquorn_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map POPQUORN 5-point offensiveness scale to the canonical 3-class schema.

    Output:
        0 = Not Hate Speech
        1 = Ambiguous / Offensive
        2 = Hate Speech
    """
    if "human_label" not in df.columns:
        return df

    def map_label(x):
        try:
            x = int(round(float(x)))
        except (ValueError, TypeError):
            return pd.NA

        if x == 1:
            return 0
        if x in (2, 3):
            return 1
        if x in (4, 5):
            return 2
        return pd.NA

    df = df.copy()
    df["human_label"] = df["human_label"].apply(map_label).astype("Int64")
    return df


def _validate_no_missing_labels(df: pd.DataFrame, dataset_key: str):
    if "human_label" not in df.columns:
        return

    missing = int(df["human_label"].isna().sum())
    if missing == 0:
        return

    raise ValueError(
        f"{dataset_key}: normalization produced {missing} missing values in 'human_label'. "
        "Stop and inspect schema/label mapping before building subsets or running the pipeline."
    )


def normalize(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    log = logging.getLogger(__name__)

    dataset_key = detectDataset(filename)
    if dataset_key is None:
        log.warning(
            "normalize(): '%s' not in DATASET_SCHEMAS — assuming canonical column names.",
            filename,
        )
        return df

    schema = DATASET_SCHEMAS[dataset_key]
    rename_map = {
        raw: canonical
        for raw, canonical in schema.items()
        if raw in df.columns and raw != canonical
    }

    if rename_map:
        df = df.rename(columns=rename_map)
        log.info("Normalized '%s': renamed %s", dataset_key, rename_map)

    if dataset_key == "measuring_hate_speech":
        df = _reconstruct_mhs_race(df)

    if dataset_key == "popquorn":
        df = _map_popquorn_labels(df)
        df = _normalize_popquorn_race(df)

    _validate_no_missing_labels(df, dataset_key)
    return df