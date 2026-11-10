"""Shared data access for Group Number 11 (Kaggle notebooks, Google Colab, or a laptop).

`ensure_data()` returns a folder that contains train.csv / unlabeled.csv / test.csv /
sample_submission.csv, with no manual steps. It looks in this order:
  1. `data_dir` (default ./data), if the files are already there
  2. /kaggle/input/...  (a Kaggle notebook with the competition attached as an input)
  3. kagglehub          (inside a Kaggle notebook this needs no credentials)
  4. the Kaggle API     (Colab / laptop; credentials from Colab Secrets, environment variables
                         KAGGLE_USERNAME + KAGGLE_KEY, or ~/.kaggle/kaggle.json)
The data is never committed to git (course rule).

`load_split()` returns the team's fixed 8,000 / 2,000 split from the committed
`train_ids.csv` / `validation_ids.csv` (stratified, random_state=42), so every experiment
uses exactly the same validation rows. No new split is ever created here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

COMPETITION = "cornell-tech-cs-5785-combined-2026-ai-vs-human-text"
TEAM_DIR = Path(__file__).resolve().parent
FILES = ["train.csv", "unlabeled.csv", "test.csv", "sample_submission.csv"]
REQUIRED = FILES[:3]
SEED = 42


def _has_data(d: Path) -> bool:
    return all((d / f).exists() for f in REQUIRED)


def _find_under(root: Path) -> Path | None:
    """Folder under `root` holding the competition CSVs (prefers the competition's own folder)."""
    if not root.exists():
        return None
    hits = [p.parent for p in root.rglob("train.csv") if _has_data(p.parent)]
    hits.sort(key=lambda d: (COMPETITION not in str(d), len(str(d))))
    return hits[0] if hits else None


def _colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name)
    except Exception:
        return None


def _setup_kaggle_credentials() -> None:
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        if not os.environ.get(name):
            val = _colab_secret(name)
            if val:
                os.environ[name] = val
    have_pair = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    token = os.environ.get("KAGGLE_API_TOKEN")
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if have_pair and not cfg.exists():  # some kaggle CLI versions insist on the file
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"username": os.environ["KAGGLE_USERNAME"],
                                   "key": os.environ["KAGGLE_KEY"]}))
        cfg.chmod(0o600)
    if not (have_pair or token or cfg.exists()):
        raise RuntimeError(
            "Competition data not found and no Kaggle credentials available.\n"
            "  Kaggle notebook: click 'Add Input' and attach the competition, then re-run.\n"
            "  Colab: add KAGGLE_USERNAME and KAGGLE_KEY under the key icon (Secrets).\n"
            "  Laptop: put the CSVs in ./data or save your token as ~/.kaggle/kaggle.json."
        )


def _download_with_api(data_dir: Path) -> None:
    _setup_kaggle_credentials()
    print(f"[data] downloading Kaggle competition '{COMPETITION}' -> {data_dir}")
    cmd = [sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION,
           "-p", str(data_dir)]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
        subprocess.run(cmd, check=True)
    for z in data_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if name in FILES:  # flatten any sub-folders inside the zip
                    (data_dir / name).write_bytes(zf.read(member))


def ensure_data(data_dir: str | Path = "data") -> Path:
    data_dir = Path(data_dir)
    if _has_data(data_dir):
        return data_dir
    found = _find_under(Path("/kaggle/input"))
    if found is None and Path("/kaggle/working").exists():
        try:  # inside a Kaggle notebook this is authenticated automatically
            import kagglehub  # type: ignore
            found = _find_under(Path(kagglehub.competition_download(COMPETITION)))
        except Exception as e:  # fall through to the API route
            print(f"[data] kagglehub route unavailable ({type(e).__name__}: {e})")
    if found is not None:
        print(f"[data] using {found}")
        return found
    data_dir.mkdir(parents=True, exist_ok=True)
    _download_with_api(data_dir)
    if not _has_data(data_dir):
        raise RuntimeError(f"Download finished but {REQUIRED} not all found in {data_dir}: "
                           f"{sorted(p.name for p in data_dir.iterdir())}")
    print("[data] ready:", sorted(p.name for p in data_dir.glob("*.csv")))
    return data_dir


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_train(data_dir: str | Path = "data") -> pd.DataFrame:
    return _read(ensure_data(data_dir) / "train.csv")


def load_split(data_dir: str | Path = "data"):
    """Return (train_df, val_df) using the committed split ID files."""
    train = load_train(data_dir).set_index("id")
    tr_ids = pd.read_csv(TEAM_DIR / "train_ids.csv")["id"]
    va_ids = pd.read_csv(TEAM_DIR / "validation_ids.csv")["id"]
    tr, va = train.loc[tr_ids].reset_index(), train.loc[va_ids].reset_index()
    assert len(tr) == 8000 and len(va) == 2000 and not set(tr.id) & set(va.id)
    print(f"[split] train={len(tr)}  val={len(va)}  val AI-rate={va.label.mean():.3f}")
    return tr, va


def load_unlabeled(data_dir: str | Path = "data") -> pd.DataFrame:
    return _read(ensure_data(data_dir) / "unlabeled.csv")


def load_test(data_dir: str | Path = "data") -> pd.DataFrame:
    return _read(ensure_data(data_dir) / "test.csv")


def load_sample_submission(data_dir: str | Path = "data") -> pd.DataFrame:
    return pd.read_csv(ensure_data(data_dir) / "sample_submission.csv")


def paired_bootstrap(y, a, b, n=1000, seed=SEED):
    """Mean and 95% CI of AUROC(b) - AUROC(a), resampling the same validation rows."""
    y, a, b = map(np.asarray, (y, a, b))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)
