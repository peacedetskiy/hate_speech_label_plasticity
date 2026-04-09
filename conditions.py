# conditions.py

import random

import pandas as pd


SUITE_BASIC = [
    "neutral",
    "original",
    "inverted",
]

SUITE_EXTENDED = [
    "dataset_random",
    "pool_random_0",
    "pool_random_1",
]

RANDOM_SEEDS = {
    "pool_random_0": 42,
    "pool_random_1": 99,
}

ACTIVE_SUITE = SUITE_BASIC


DEMOGRAPHIC_POOL = {
    "age": [18, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "gender": ["Man", "Woman", "Non-binary"],
    "race": [
        "White",
        "Black or African American",
        "Hispanic or Latino",
        "Asian",
        "Middle Eastern",
        "Native American",
        "Mixed / Other",
    ],
}


RACE_INVERSION = {
    "White": "Black or African American",
    "Black or African American": "White",
    "Hispanic or Latino": "White",
    "Asian": "White",
    "Middle Eastern": "White",
    "Native American": "White",
    "Mixed / Other": "White",
}


def invertGender(gender: str) -> str:
    g = str(gender).strip().lower()
    if g in ("man", "male", "m"):
        return "Woman"
    if g in ("woman", "female", "f"):
        return "Man"
    return "Man"


def _parse_age_for_inversion(age):
    """
    Supports:
    - numeric ages (e.g. 34, 34.0, "34")
    - age ranges (e.g. "35-39", "18-24"), converted to midpoint
    """
    if isinstance(age, str):
        age = age.strip()
        if "-" in age:
            left, right = age.split("-", 1)
            left = int(float(left.strip()))
            right = int(float(right.strip()))
            return (left + right) // 2
    return int(float(age))


def invertAge(age) -> int:
    try:
        age = _parse_age_for_inversion(age)
    except (ValueError, TypeError):
        return 22
    return 65 if age < 35 else 22


def invertRace(race: str) -> str:
    return RACE_INVERSION.get(str(race).strip(), "White")


def poolPersona(row_index: int, seed: int, original_race: str = "") -> dict:
    rng = random.Random(row_index + seed)
    age = rng.choice(DEMOGRAPHIC_POOL["age"])
    gender = rng.choice(DEMOGRAPHIC_POOL["gender"])

    race_pool = [r for r in DEMOGRAPHIC_POOL["race"] if r != str(original_race).strip()]
    race = rng.choice(race_pool if race_pool else DEMOGRAPHIC_POOL["race"])

    return {"age": age, "gender": gender, "race": race}


def sampleDatasetPersonas(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    indices = list(df.index)

    has_race = "annotator_race" in df.columns
    has_age = "annotator_age" in df.columns
    has_gender = "annotator_gender" in df.columns

    dr_age, dr_gender, dr_race = [], [], []

    for idx in indices:
        orig_race = str(df.at[idx, "annotator_race"]).strip() if has_race else ""

        candidates = [
            i
            for i in indices
            if i != idx and (not has_race or str(df.at[i, "annotator_race"]).strip() != orig_race)
        ]
        if not candidates:
            candidates = [i for i in indices if i != idx]

        chosen = rng.choice(candidates) if candidates else idx

        dr_age.append(df.at[chosen, "annotator_age"] if has_age else "unknown")
        dr_gender.append(df.at[chosen, "annotator_gender"] if has_gender else "unknown")
        dr_race.append(df.at[chosen, "annotator_race"] if has_race else "unknown")

    df = df.copy()
    df["_dr_age"] = dr_age
    df["_dr_gender"] = dr_gender
    df["_dr_race"] = dr_race
    return df


ONESHOT_STATEMENT = "The weather today is really nice."
ONESHOT_LABEL = "0"


def _responseInstruction() -> str:
    return (
        "Respond only with one character:\n"
        "0 = Not Hate Speech\n"
        "1 = Hate Speech"
    )


def _neutralPrompt(statement: str) -> str:
    return (
        "You are acting as a data annotator.\n\n"
        f'Example:\nStatement: "{ONESHOT_STATEMENT}"\nLabel: {ONESHOT_LABEL}\n\n'
        "Task:\nLabel the following statement.\n"
        f'Statement: "{statement}"\n'
        f"{_responseInstruction()}"
    )


def _personaPrompt(statement: str, age, gender: str, race: str) -> str:
    return (
        "You are acting as a data annotator.\n"
        "Your background:\n"
        f"Age: {age}\n"
        f"Gender: {gender}\n"
        f"Nationality/Race: {race}\n\n"
        f'Example:\nStatement: "{ONESHOT_STATEMENT}"\nLabel: {ONESHOT_LABEL}\n\n'
        "Task:\nAs this same person, label the following statement.\n"
        f'Statement: "{statement}"\n'
        f"{_responseInstruction()}"
    )


def buildPersonaPrompt(statement: str, age, gender: str, race: str) -> str:
    return _personaPrompt(statement, age, gender, race)


def buildPrompt(row: pd.Series, condition: str) -> str:
    statement = str(row["text"])

    if condition == "neutral":
        return _neutralPrompt(statement)

    if condition == "original":
        return _personaPrompt(
            statement,
            row.get("annotator_age", "unknown"),
            row.get("annotator_gender", "unknown"),
            row.get("annotator_race", "unknown"),
        )

    if condition == "dataset_random":
        return _personaPrompt(
            statement,
            row.get("_dr_age", "unknown"),
            row.get("_dr_gender", "unknown"),
            row.get("_dr_race", "unknown"),
        )

    if condition == "inverted":
        return _personaPrompt(
            statement,
            invertAge(row.get("annotator_age", 35)),
            invertGender(row.get("annotator_gender", "Man")),
            invertRace(row.get("annotator_race", "White")),
        )

    if condition in RANDOM_SEEDS:
        persona = poolPersona(
            int(row.name),
            RANDOM_SEEDS[condition],
            original_race=str(row.get("annotator_race", "")),
        )
        return _personaPrompt(statement, persona["age"], persona["gender"], persona["race"])

    raise ValueError(f"buildPrompt(): unknown condition '{condition}'")