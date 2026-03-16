import pandas as pd


SUITE_FIRST_EXPERIMENT = [
    "neutral",
    "original",
    "inverted",
]

ACTIVE_SUITE = SUITE_FIRST_EXPERIMENT


RACE_INVERSION = {
    "White": "Black or African American",
    "Black or African American": "White",
    "Hispanic or Latino": "White",
    "Asian": "White",
    "Middle Eastern": "White",
    "Native American": "White",
    "Mixed / Other": "White",
}


ONESHOT_STATEMENT = "The weather today is really nice."
ONESHOT_LABEL = "0"


def invertGender(gender: str) -> str:
    g = str(gender).strip().lower()
    if g in ("man", "male", "m"):
        return "Woman"
    if g in ("woman", "female", "f"):
        return "Man"
    return "Man"


def invertAge(age) -> int:
    try:
        age = int(float(age))
    except (ValueError, TypeError):
        return 22
    return 65 if age < 35 else 22


def invertRace(race: str) -> str:
    return RACE_INVERSION.get(str(race).strip(), "White")


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

    if condition == "inverted":
        return _personaPrompt(
            statement,
            invertAge(row.get("annotator_age", 35)),
            invertGender(row.get("annotator_gender", "Man")),
            invertRace(row.get("annotator_race", "White")),
        )

    raise ValueError(f"buildPrompt(): unknown condition '{condition}'")