from datasets import load_dataset
import pandas as pd
from pathlib import Path
import requests
import zipfile
import io
import json

DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------
# 1) Measuring Hate Speech
# ----------------------------
print("Loading Measuring Hate Speech from Hugging Face...")
ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")
df = pd.DataFrame(ds["train"])

# Align with your pipeline if needed
if "text" in df.columns and "title" not in df.columns:
    df = df.rename(columns={"text": "title"})

df.to_parquet(DATA_DIR / "measuring_hate_speech.parquet", index=False)
print("Saved datasets/measuring_hate_speech.parquet")


# ----------------------------
# 2) POPQUORN from GitHub
# ----------------------------
POPQUORN_ZIP_URL = "https://codeload.github.com/Jiaxin-Pei/Potato-Prolific-Dataset/zip/refs/heads/main"


def download_and_extract_github_zip(url: str, extract_to: Path) -> Path:
    print("Downloading POPQUORN from GitHub...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(extract_to)

    # GitHub archives usually extract into something like Potato-Prolific-Dataset-main/
    extracted_roots = [p for p in extract_to.iterdir() if p.is_dir()]
    if not extracted_roots:
        raise FileNotFoundError("No extracted repository folder found after download.")

    # Pick the most likely repo root
    repo_root = max(extracted_roots, key=lambda p: len(list(p.rglob("*"))))
    return repo_root


def load_table_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    elif suffix in {".jsonl", ".json"}:
        # Try JSONL first, then standard JSON
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                # common patterns
                for key in ["data", "examples", "records", "items"]:
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])
                return pd.json_normalize(data)
            raise ValueError(f"Unsupported JSON structure in {path}")
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def pick_best_text_column(df: pd.DataFrame) -> pd.DataFrame:
    # Try likely text fields in order
    candidate_cols = [
        "title",
        "text",
        "comment_text",
        "comment",
        "post",
        "content",
        "sentence",
        "utterance",
        "question",
        "source_text",
        "input",
        "document",
    ]

    for col in candidate_cols:
        if col in df.columns:
            if col != "title":
                df = df.rename(columns={col: "title"})
            return df

    return df  # leave untouched if nothing obvious exists


def load_popquorn_offensiveness() -> pd.DataFrame:
    temp_root = DATA_DIR / "_tmp_popquorn"
    temp_root.mkdir(exist_ok=True)

    repo_root = download_and_extract_github_zip(POPQUORN_ZIP_URL, temp_root)

    dataset_root = repo_root / "dataset"
    if not dataset_root.exists():
        raise FileNotFoundError(f"'dataset' folder not found in {repo_root}")

    offensiveness_dir = dataset_root / "offensiveness"
    search_root = offensiveness_dir if offensiveness_dir.exists() else dataset_root

    # Search for likely tabular files
    candidates = []
    for ext in ("*.csv", "*.tsv", "*.jsonl", "*.json", "*.parquet"):
        candidates.extend(search_root.rglob(ext))

    if not candidates:
        raise FileNotFoundError(f"No supported data files found in {search_root}")

    # Prefer filenames/directories that look like offensiveness data
    def score(path: Path) -> tuple[int, int]:
        name = str(path).lower()
        priority = 0
        if "offens" in name:
            priority += 10
        if "annot" in name:
            priority += 2
        if "data" in name:
            priority += 1
        return (priority, -len(path.name))

    candidates = sorted(candidates, key=score, reverse=True)

    last_error = None
    for file_path in candidates:
        try:
            print(f"Trying POPQUORN file: {file_path}")
            df = load_table_file(file_path)
            if len(df) == 0:
                continue
            df = pick_best_text_column(df)
            return df
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Could not load any POPQUORN file successfully. Last error: {last_error}")


print("Loading POPQUORN from GitHub...")
pop_df = load_popquorn_offensiveness()

# Align with your pipeline
pop_df = pick_best_text_column(pop_df)

pop_df.to_parquet(DATA_DIR / "popquorn.parquet", index=False)
print("Saved datasets/popquorn.parquet")

print("\nDone.")