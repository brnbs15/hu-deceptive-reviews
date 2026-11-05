"""Közös függvények, amiket a többi script is használ
(elérési utak, seed, adatbetöltés, foldok, metrikák, eredmények mentése)."""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

# A projekt gyökérmappája (az src mappa "szülője")
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "deceptive-opinion.csv"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

SEED = 42
N_FOLDS = 5

# A címkéket számokká alakítjuk: 0 = valódi, 1 = megtévesztő
LABEL2ID = {"truthful": 0, "deceptive": 1}
ID2LABEL = {0: "truthful", 1: "deceptive"}


def set_seed(seed=SEED):
    # Minden véletlenszám-generátornak ugyanazt a seedet adjuk,
    # hogy újrafuttatáskor is ugyanazok az eredmények jöjjenek ki.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        # a baseline-hoz nem kell torch, ilyenkor nem baj ha nincs telepítve
        pass


def load_raw():
    # Az eredeti angol adathalmaz (Ott et al.) betöltése
    df = pd.read_csv(DATA_RAW)
    df = df.rename(columns={"text": "text_en"})
    # minden véleménynek adunk egy sorszámot, hogy később vissza lehessen keresni
    df.insert(0, "id", np.arange(len(df)))
    df["label"] = df["deceptive"].map(LABEL2ID)
    df["text_en"] = df["text_en"].astype(str).str.strip()
    return df


def load_hu():
    # A lefordított magyar adathalmaz betöltése (ezt a translate.py csinálja meg)
    path = DATA_PROC / "reviews_hu.csv"
    if not path.exists():
        raise FileNotFoundError(f"Nincs meg a {path} fájl, előbb futtasd: python -m src.translate")
    return pd.read_csv(path)


def make_folds(df):
    """Kiosztja a véleményeket 5 foldba a keresztvalidációhoz.

    Fontos: egy szálloda összes véleménye ugyanabba a foldba kerül
    (20 szálloda van, így foldonként 4). Különben a modell megtanulhatná
    a szállodák nevét / jellemző szavait, és jobbnak tűnne, mint amilyen.
    Ott et al. (2011) is így csinálta.
    """
    path = DATA_PROC / "folds.csv"

    # Ha már egyszer elkészült a beosztás, azt használjuk, hogy minden modell
    # biztosan ugyanazokon a foldokon fusson.
    if path.exists():
        saved = pd.read_csv(path)
        return saved.set_index("id").loc[df["id"], "fold"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.full(len(df), -1)
    splits = splitter.split(df, df["label"], groups=df["hotel"])
    for k, (train_idx, test_idx) in enumerate(splits):
        folds[test_idx] = k

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], "fold": folds}).to_csv(path, index=False)
    return folds


def metrics(y_true, y_pred):
    # Pontosság és makro-F1 (a makro-F1 a két osztály F1-ének sima átlaga)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": float(acc), "macro_f1": float(f1)}


def save_oof(name, df, folds, prob, extra=None):
    """Elmenti a predikciókat és a metrikákat.

    Az "oof" = out-of-fold, vagyis minden véleményre az a predikció,
    amit akkor kapott, amikor az ő foldja volt a teszthalmaz.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame({
        "id": df["id"],
        "fold": folds,
        "label": df["label"],
        "polarity": df["polarity"],
        "prob_deceptive": prob,
        "pred": (prob >= 0.5).astype(int),  # 0.5 felett megtévesztőnek mondjuk
    })
    out.to_csv(RESULTS / f"oof_{name}.csv", index=False)

    # metrikák foldonként
    per_fold = []
    for k, group in out.groupby("fold"):
        m = metrics(group["label"], group["pred"])
        m["fold"] = int(k)
        per_fold.append(m)

    accs = [p["accuracy"] for p in per_fold]
    f1s = [p["macro_f1"] for p in per_fold]

    summary = {
        "model": name,
        "per_fold": per_fold,
        "mean": {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))},
        "std": {"accuracy": float(np.std(accs, ddof=1)), "macro_f1": float(np.std(f1s, ddof=1))},
        # az összes predikción egyben számolva (nem foldonként átlagolva)
        "pooled": metrics(out["label"], out["pred"]),
    }
    if extra:
        summary["config"] = extra  # a beállítások is legyenek elmentve

    with open(RESULTS / f"metrics_{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[{name}] pontosság = {summary['mean']['accuracy']:.4f} ± {summary['std']['accuracy']:.4f} | "
          f"makro-F1 = {summary['mean']['macro_f1']:.4f} ± {summary['std']['macro_f1']:.4f}")
