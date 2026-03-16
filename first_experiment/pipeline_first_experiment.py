import logging
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from schema_first_experiment import normalize
from conditions_first_experiment import ACTIVE_SUITE, buildPrompt


os.makedirs("logs", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logs/ollama.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

CONNECT_TIMEOUT = 10
QUERY_TIMEOUT = 180
PULL_TIMEOUT = 3600

PROCESSED_LOG = "logs/processedPapers.log"
ERROR_LOG = "logs/modelsErrors.log"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

MODE = os.environ.get("PIPELINE_MODE", "debug").strip().lower()
if MODE not in {"debug", "batch"}:
    MODE = "debug"

DEBUG_MODEL = os.environ.get("PIPELINE_DEBUG_MODEL", "gemma2:2b")
OVERWRITE_EXISTING = os.environ.get("PIPELINE_OVERWRITE", "0").strip() == "1"

DEFAULT_WORKERS = 1 if MODE == "debug" else 2
DEFAULT_RETRIES = 5 if MODE == "debug" else 2
DEFAULT_CHECKPOINT_EVERY = 10 if MODE == "debug" else 25

MAX_WORKERS = max(1, int(os.environ.get("PIPELINE_WORKERS", str(DEFAULT_WORKERS))))
MAX_RETRIES = max(1, int(os.environ.get("PIPELINE_MAX_RETRIES", str(DEFAULT_RETRIES))))
CHECKPOINT_EVERY = max(1, int(os.environ.get("PIPELINE_CHECKPOINT_EVERY", str(DEFAULT_CHECKPOINT_EVERY))))

OLLAMA_OPTIONS = {
    "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0")),
    "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "4")),
    "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "384")),
}

if os.environ.get("OLLAMA_USE_STOP_NEWLINE", "1").strip() == "1":
    OLLAMA_OPTIONS["stop"] = ["\n"]

OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "2m")
DEFAULT_DATASET = "../datasets/measuring_hate_speech_test10.parquet"

_THREAD_LOCAL = threading.local()


def _build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.headers.update({"Content-Type": "application/json"})
    return session


def _get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = _build_session()
        _THREAD_LOCAL.session = session
    return session


def _ollama_url(path: str) -> str:
    return f"{OLLAMA_BASE_URL}{path}"


def _log_error_line(message: str):
    with open(ERROR_LOG, "a+", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def _safe_json(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except ValueError:
        return {}


def _ollama_alive() -> bool:
    try:
        resp = _get_session().get(
            _ollama_url("/api/version"),
            timeout=(CONNECT_TIMEOUT, CONNECT_TIMEOUT),
        )
        return resp.ok
    except requests.RequestException as e:
        logger.error("Ollama health check failed: %s", e)
        return False


def _require_ollama_service() -> bool:
    if _ollama_alive():
        return True

    msg = (
        f"Could not reach local Ollama service at {OLLAMA_BASE_URL}. "
        "Make sure Ollama is running before starting the pipeline."
    )
    logger.error(msg)
    _log_error_line(msg)
    return False


def _get_installed_models() -> list[str]:
    try:
        resp = _get_session().get(
            _ollama_url("/api/tags"),
            timeout=(CONNECT_TIMEOUT, CONNECT_TIMEOUT),
        )
        resp.raise_for_status()
        data = _safe_json(resp)
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except requests.RequestException as e:
        logger.error("Failed to read installed models from Ollama: %s", e)
        return []


def _model_installed(model: str) -> bool:
    target = model.strip().lower()
    for name in _get_installed_models():
        normalized_name = name.strip().lower()
        if normalized_name == target or normalized_name.split(":")[0] == target:
            return True
    return False


def _pull_model(model: str) -> bool:
    try:
        resp = _get_session().post(
            _ollama_url("/api/pull"),
            json={"name": model, "stream": False},
            timeout=(CONNECT_TIMEOUT, PULL_TIMEOUT),
        )
        resp.raise_for_status()
        _ = _safe_json(resp)
        return True
    except requests.RequestException as e:
        logger.error("Model pull failed for '%s': %s", model, e)
        _log_error_line(f"{model} - install failed via HTTP API: {e}")
        return False


def _warmup_model(model: str) -> bool:
    try:
        resp = _get_session().post(
            _ollama_url("/api/generate"),
            json={
                "model": model,
                "prompt": "Reply only with 0",
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": 0,
                    "num_predict": 2,
                    "num_ctx": 64,
                    "stop": ["\n"],
                },
            },
            timeout=(CONNECT_TIMEOUT, 120),
        )
        resp.raise_for_status()
        data = _safe_json(resp)
        return bool(str(data.get("response", "")).strip())
    except requests.RequestException as e:
        logger.error("Warm-up failed for '%s': %s", model, e)
        _log_error_line(f"{model} - warmup failed: {e}")
        return False


def _normalize_label_output(output: str) -> str:
    if output is None:
        return ""

    raw = str(output).strip()
    if not raw:
        return ""

    compact = " ".join(raw.split())
    lowered = compact.lower()

    if re.fullmatch(r"0[\.\)]?", compact):
        return "Not Hate Speech"
    if re.fullmatch(r"1[\.\)]?", compact):
        return "Hate Speech"

    if "not hate speech" in lowered or "non-hate speech" in lowered:
        return "Not Hate Speech"
    if lowered.startswith("0 "):
        return "Not Hate Speech"
    if lowered in {"0", "0.", "0)"}:
        return "Not Hate Speech"

    if lowered.startswith("hate speech") or lowered == "hate speech":
        return "Hate Speech"
    if lowered.startswith("1 "):
        return "Hate Speech"
    if lowered in {"1", "1.", "1)"}:
        return "Hate Speech"

    return ""


def _is_filled(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _column_complete(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    return df[col].map(_is_filled).all()


def _write_dataset(df: pd.DataFrame, filename: str):
    tmp_path = f"{filename}.tmp"
    df.to_parquet(tmp_path)
    os.replace(tmp_path, filename)


def queryOllama(query: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": query,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": OLLAMA_OPTIONS,
    }

    try:
        resp = _get_session().post(
            _ollama_url("/api/generate"),
            json=payload,
            timeout=(CONNECT_TIMEOUT, QUERY_TIMEOUT),
        )
        resp.raise_for_status()
        data = _safe_json(resp)
    except requests.RequestException as e:
        logger.error("Generation failed | model=%s | error=%s", model, e)
        _log_error_line(f"{model} | query: {query[:80]} | error: {e}")
        return ""

    raw_response = str(data.get("response", "")).strip()
    normalized = _normalize_label_output(raw_response)
    if normalized:
        return normalized

    err = data.get("error", f"unparseable response: {raw_response[:120]}")
    logger.error("Bad generation response | model=%s | error=%s", model, err)
    _log_error_line(f"{model} | query: {query[:80]} | error: {err}")
    return ""


def installModelIfNotExist(model: str) -> bool:
    if not _require_ollama_service():
        return False

    if not _model_installed(model):
        logger.info("Model '%s' not found locally. Pulling via Ollama HTTP API...", model)
        if not _pull_model(model):
            return False

    if not _model_installed(model):
        logger.error("Model '%s' is still not listed after pull.", model)
        _log_error_line(f"{model} - install verification failed")
        return False

    logger.info("Model '%s' is installed.", model)

    if not _warmup_model(model):
        logger.warning("Model '%s' installed, but warm-up failed.", model)

    return True


def removeModelFromDisk(model: str):
    if sys.platform in ("linux", "linux2"):
        directory_path = "/usr/share/ollama/.ollama"
    elif sys.platform == "darwin":
        directory_path = os.path.expanduser("~/.ollama")
    elif sys.platform == "win32":
        username = os.environ.get("USERNAME") or os.getlogin()
        directory_path = f"C:/Users/{username}/.ollama"
    else:
        sys.exit(f"Unsupported OS: {sys.platform}")

    shutil.rmtree(directory_path, ignore_errors=True)
    os.makedirs(directory_path, exist_ok=True)


def getModelsFromOllama() -> list[str]:
    if not _require_ollama_service():
        return []
    return sorted(_get_installed_models())


def getProcessedEntries() -> set[str]:
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def addProcessedEntry(entry: str):
    with open(PROCESSED_LOG, "a+", encoding="utf-8") as f:
        f.write(entry + "\n")


def runExperiment(row: pd.Series, model: str, condition: str) -> str:
    prompt = buildPrompt(row, condition)

    for attempt in range(1, MAX_RETRIES + 1):
        output = queryOllama(prompt, model)
        if output:
            return output

        logger.warning(
            "Empty response | model=%s | condition=%s | attempt=%d/%d | text='%s'",
            model,
            condition,
            attempt,
            MAX_RETRIES,
            str(row["text"])[:60],
        )

        if attempt < MAX_RETRIES:
            time.sleep(min(attempt * 2, 8))

    logger.error(
        "All %d attempts failed | model=%s | condition=%s | text='%s'",
        MAX_RETRIES,
        model,
        condition,
        str(row["text"])[:60],
    )
    return ""


def _run_condition(
    df: pd.DataFrame,
    filename: str,
    model: str,
    condition: str,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, bool]:
    dataset_name = os.path.basename(filename)
    col = f"{model}_{condition}"

    if col not in df.columns:
        df[col] = ""

    pending_indices = [
        idx for idx in df.index
        if overwrite or not _is_filled(df.at[idx, col])
    ]

    if not pending_indices:
        logger.info("Skipping completed column: %s | %s", dataset_name, col)
        return df, False

    changed = False
    completed_since_save = 0
    skipped_count = len(df) - len(pending_indices)

    desc = f"{model} | {dataset_name} | {condition}"
    logger.info(
        "Starting condition | model=%s | dataset=%s | condition=%s | pending=%d | skipped=%d | workers=%d",
        model,
        dataset_name,
        condition,
        len(pending_indices),
        skipped_count,
        MAX_WORKERS,
    )

    progress = tqdm(
        total=len(pending_indices),
        desc=desc,
        leave=False,
        ascii=True,
        dynamic_ncols=True,
    )
    progress.set_postfix_str(f"done={skipped_count}/{len(df)}")

    def _task(idx):
        row = df.loc[idx].copy()
        return idx, runExperiment(row, model, condition)

    if MAX_WORKERS == 1:
        iterator = (_task(idx) for idx in pending_indices)
        for idx, output in iterator:
            df.at[idx, col] = output
            changed = True
            completed_since_save += 1
            progress.update(1)
            progress.set_postfix_str(f"done={skipped_count + progress.n}/{len(df)}")

            if completed_since_save >= CHECKPOINT_EVERY:
                _write_dataset(df, filename)
                completed_since_save = 0
                logger.info(
                    "Checkpoint saved | model=%s | dataset=%s | condition=%s | completed=%d/%d",
                    model,
                    dataset_name,
                    condition,
                    skipped_count + progress.n,
                    len(df),
                )
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(_task, idx): idx
                for idx in pending_indices
            }

            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    idx, output = future.result()
                except Exception as e:
                    logger.exception(
                        "Unhandled worker failure | model=%s | dataset=%s | condition=%s | idx=%s | error=%s",
                        model,
                        dataset_name,
                        condition,
                        idx,
                        e,
                    )
                    output = ""

                df.at[idx, col] = output
                changed = True
                completed_since_save += 1
                progress.update(1)
                progress.set_postfix_str(f"done={skipped_count + progress.n}/{len(df)}")

                if completed_since_save >= CHECKPOINT_EVERY:
                    _write_dataset(df, filename)
                    completed_since_save = 0
                    logger.info(
                        "Checkpoint saved | model=%s | dataset=%s | condition=%s | completed=%d/%d",
                        model,
                        dataset_name,
                        condition,
                        skipped_count + progress.n,
                        len(df),
                    )

    progress.close()

    if changed:
        _write_dataset(df, filename)

    logger.info(
        "Condition complete | model=%s | dataset=%s | condition=%s | rows_written=%d",
        model,
        dataset_name,
        condition,
        len(pending_indices),
    )
    return df, changed


def main(
    model: str,
    suite: list = None,
    overwrite: bool = OVERWRITE_EXISTING,
    dataset_path: str = DEFAULT_DATASET,
):
    if suite is None:
        suite = ACTIVE_SUITE

    if not _require_ollama_service():
        sys.exit("Local Ollama service is not reachable. Exiting.")

    processed = getProcessedEntries()

    logger.info(
        "Pipeline start | model=%s | mode=%s | workers=%d | retries=%d | checkpoint_every=%d | overwrite=%s",
        model,
        MODE,
        MAX_WORKERS,
        MAX_RETRIES,
        CHECKPOINT_EVERY,
        overwrite,
    )

    filename = dataset_path
    logger.info("Loading dataset: %s", filename)

    df = normalize(pd.read_parquet(filename), filename)
    if "text" not in df.columns:
        raise KeyError(
            f"{os.path.basename(filename)} is missing required column 'text' after normalize(). "
            f"Available columns: {list(df.columns)}"
        )

    changed_any = False

    for condition in suite:
        col = f"{model}_{condition}"
        key = f"{model}::{filename}::{condition}"

        if not overwrite and key in processed and _column_complete(df, col):
            logger.info("Fast-skip processed condition: %s", key)
            continue

        if not overwrite and _column_complete(df, col):
            logger.info("Fast-skip full existing column: %s", col)
            if key not in processed:
                addProcessedEntry(key)
                processed.add(key)
            continue

        df, changed = _run_condition(
            df=df,
            filename=filename,
            model=model,
            condition=condition,
            overwrite=overwrite,
        )
        changed_any = changed_any or changed

        if _column_complete(df, col) and key not in processed:
            addProcessedEntry(key)
            processed.add(key)
            logger.info("Completed: %s", key)

    if changed_any:
        _write_dataset(df, filename)
        logger.info("Dataset saved: %s", filename)

    logger.info("Pipeline finished for model: %s", model)


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET

    if MODE == "debug":
        models = [DEBUG_MODEL]
    else:
        models = getModelsFromOllama()
        if not models:
            sys.exit("Could not retrieve local model list from Ollama. Exiting.")

    for model in models:
        print(
            f"\nRunning model={model} | mode={MODE} | workers={MAX_WORKERS} | "
            f"retries={MAX_RETRIES} | checkpoint_every={CHECKPOINT_EVERY} | overwrite={OVERWRITE_EXISTING}"
        )

        if not installModelIfNotExist(model):
            logger.error("Skipping model '%s' — could not install or reach Ollama.", model)
            continue

        main(model, dataset_path=dataset_path)

        if MODE != "debug":
            removeModelFromDisk(model)