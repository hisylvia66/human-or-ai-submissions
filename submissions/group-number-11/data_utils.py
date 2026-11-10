"""Shared data access for Group Number 11 (works locally and on Google Colab).

`ensure_data()` makes sure train.csv / unlabeled.csv / test.csv / sample_submission.csv are in
`data_dir`. If they are missing it downloads the competition from Kaggle automatically, so a
Colab run needs no manual uploads. The data is never committed to git (course rule).

Kaggle credentials are looked up in this order (set up once, then every run is automatic):
  1. Colab Secrets  KAGGLE_USERNAME + KAGGLE_KEY   (or KAGGLE_API_TOKEN)
  2. Environment variables with the same names
  3. ~/.kaggle/kaggle.json
You must also have clicked "Join Competition" / accepted the rules on Kaggle once.

`load_split()` returns the team's fixed 8,000 / 2,000 split using the committed
`train_ids.csv` / `validation_ids.csv` (stratified, random_state=42), so every experiment
uses exactly the same validation rows.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

COMPETITION = "cornell-tech-cs-5785-combined-2026-ai-vs-human-text"
TEAM_DIR = Path(__file__).resolve().parent
FILES = ["train.csv", "unlabeled.csv", "test.csv", "sample_submission.csv"]
SEED = 42


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
            "No Kaggle credentials found. One-time setup: kaggle.com -> Settings -> API -> "
            "'Create New Token'. In Colab, open the key icon (Secrets) and add KAGGLE_USERNAME "
            "and KAGGLE_KEY (from kaggle.json), with notebook access switched on."
        )


def ensure_data(data_dir: str | Path = "data") -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / f).exists() for f in FILES[:3]):
        return data_dir
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
    missing = [f for f in FILES[:3] if not (data_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Download finished but {missing} not found in {data_dir}: "
                           f"{sorted(p.name for p in data_dir.iterdir())}")
    print("[data] ready:", sorted(p.name for p in data_dir.glob("*.csv")))
    return data_dir


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_split(data_dir: str | Path = "data"):
    """Return (train_df, val_df) using the committed split ID files."""
    data_dir = ensure_data(data_dir)
    train = _read(data_dir / "train.csv")
    tr_ids = pd.read_csv(TEAM_DIR / "train_ids.csv")["id"]
    va_ids = pd.read_csv(TEAM_DIR / "validation_ids.csv")["id"]
    tr = train.set_index("id").loc[tr_ids].reset_index()
    va = train.set_index("id").loc[va_ids].reset_index()
    assert len(tr) == 8000 and len(va) == 2000 and not set(tr.id) & set(va.id)
    print(f"[split] train={len(tr)}  val={len(va)}  val AI-rate={va.label.mean():.3f}")
    return tr, va


def load_unlabeled(data_dir: str | Path = "data") -> pd.DataFrame:
    return _read(ensure_data(data_dir) / "unlabeled.csv")


def load_test(data_dir: str | Path = "data") -> pd.DataFrame:
    return _read(ensure_data(data_dir) / "test.csv")
