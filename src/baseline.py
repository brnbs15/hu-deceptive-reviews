"""Alapmodell (baseline): TF-IDF + logisztikus regresszió.

    python -m src.baseline --lang hu   # a magyar fordításon (ez a fő összehasonlítás)
    python -m src.baseline --lang en   # az angol eredetin (hogy lássuk, mennyit ront a fordítás)

A C paramétert (regularizáció erőssége) minden foldnál csak a tanító adaton
belül választjuk ki egy kisebb, 3-as keresztvalidációval, így a teszt fold
nem "szivárog be" a paraméterválasztásba.
"""

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from .common import RESULTS, SEED, load_hu, load_raw, make_folds, save_oof, set_seed

# ezeket a C értékeket próbáljuk ki
C_VALUES = [0.1, 0.3, 1, 3, 10, 30, 100]


def build_pipeline():
    # TF-IDF: szavak és szópárok (1-2 gramok), legalább 2 dokumentumban kell előfordulniuk
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",  # az 1 betűs szavakat is megtartjuk
    )
    lr = LogisticRegression(max_iter=5000, solver="liblinear", random_state=SEED)
    return Pipeline([("tfidf", tfidf), ("lr", lr)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["hu", "en"], default="hu")
    args = parser.parse_args()
    set_seed()

    if args.lang == "hu":
        df = load_hu()
    else:
        df = load_raw()
    text_col = "text_" + args.lang

    folds = make_folds(df)
    n_folds = folds.max() + 1  # a foldok 0-tól vannak számozva
    prob = np.zeros(len(df))
    best_c_values = []

    for k in range(n_folds):
        train = folds != k
        test = folds == k

        # C kiválasztása a tanító részen belül (itt is szállodák szerint csoportosítva)
        search = GridSearchCV(
            build_pipeline(),
            {"lr__C": C_VALUES},
            scoring="f1_macro",
            cv=GroupKFold(n_splits=3),
            n_jobs=-1,
        )
        search.fit(df.loc[train, text_col], df.loc[train, "label"], groups=df.loc[train, "hotel"])
        best_c_values.append(search.best_params_["lr__C"])

        # a legjobb modellel jósolunk a teszt foldra
        prob[test] = search.predict_proba(df.loc[test, text_col])[:, 1]
        print(f"  fold {k}: legjobb C = {search.best_params_['lr__C']}")

    config = {"ngram_range": [1, 2], "min_df": 2, "C_grid": C_VALUES, "chosen_C_per_fold": best_c_values}
    save_oof(f"tfidf_lr_{args.lang}", df, folds, prob, extra=config)

    # Melyik szavak a legjellemzőbbek a két osztályra?
    # Ehhez egy modellt tanítunk a teljes adaton (a foldokon választott C-k mediánjával),
    # és megnézzük a legnagyobb / legkisebb súlyú szavakat.
    pipe = build_pipeline()
    pipe.set_params(lr__C=float(np.median(best_c_values)))
    pipe.fit(df[text_col], df["label"])

    words = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    weights = pipe.named_steps["lr"].coef_[0]
    order = np.argsort(weights)  # növekvő sorrend: elején a "valódi", végén a "megtévesztő" szavak

    top = {
        "deceptive": words[order[::-1][:25]].tolist(),
        "truthful": words[order[:25]].tolist(),
    }
    with open(RESULTS / f"top_features_{args.lang}.json", "w", encoding="utf-8") as f:
        json.dump(top, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
