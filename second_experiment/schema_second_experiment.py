import logging

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
        "post": CANONICAL["text"],
        "offensiveness": CANONICAL["label"],
        "age": CANONICAL["age"],
        "gender": CANONICAL["gender"],
        "race": CANONICAL["race"],
        "worker_id": CANONICAL["ann_id"],
    },
}


def detectDataset(filename: str) -> str | None:
    base = filename.lower().replace("-", "_").replace(" ", "_")
    for key in DATASET_SCHEMAS:
        if key in base:
            return key
    return None


def _reconstruct_mhs_race(df):
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


def normalize(df, filename: str):
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

    return df