"""Az eredmények összesítése: táblázat, McNemar-próba, bontás polaritás szerint, ábrák.

    python -m src.analyze

Kimenet: results/results.json (minden szám egy helyen) és results/fig_*.png (ábrák).
"""

import json

import matplotlib
matplotlib.use("Agg")  # ábrák mentése fájlba, ablak megnyitása nélkül
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

from .common import DATA_PROC, RESULTS, metrics

# (fájlnév, név az ábrákon)
MODELS = [
    ("tfidf_lr_hu", "TF-IDF + LR (magyar)"),
    ("hubert", "huBERT finomhangolva"),
    ("hybrid", "huBERT + CNN–BiLSTM"),
    ("tfidf_lr_en", "TF-IDF + LR (angol eredeti)"),
]
COLORS = {"tfidf_lr_hu": "#9AA5B1", "hubert": "#1F6FB2", "hybrid": "#E07A1F", "tfidf_lr_en": "#D5DBE1"}


def mcnemar(y, pred_a, pred_b):
    """McNemar-próba: szignifikánsan jobb-e az egyik modell a másiknál?

    Csak azok a vélemények számítanak, ahol a két modell nem ért egyet
    (az egyik eltalálta, a másik nem). Ha a két modell egyforma jó,
    akkor ezek nagyjából 50-50 arányban oszlanak meg -> binomiális próba.
    """
    a_correct = pred_a == y
    b_correct = pred_b == y
    only_a = int((a_correct & ~b_correct).sum())
    only_b = int((~a_correct & b_correct).sum())

    if only_a + only_b == 0:
        p = 1.0  # sehol nem tértek el
    else:
        p = binomtest(only_a, only_a + only_b, 0.5).pvalue
    return {"a_only_correct": only_a, "b_only_correct": only_b, "p_value": float(p)}


def mt_quality():
    # A kézzel kitöltött fordítás-értékelés összesítése (ha van)
    path = DATA_PROC / "mt_quality_sample.csv"
    if not path.exists():
        return None
    sample = pd.read_csv(path)
    result = {"n_sample": int(len(sample))}

    for col in ["adequacy_1_5", "fluency_1_5"]:
        values = pd.to_numeric(sample[col], errors="coerce").dropna()  # üres cellák kihagyása
        result[col] = {
            "n_rated": int(len(values)),
            "mean": float(values.mean()) if len(values) > 0 else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        }

    cue = sample["cue_preserved_i_n"].astype(str).str.strip().str.lower()
    rated = cue.isin(["i", "n", "igen", "nem"])
    if rated.any():
        result["cue_preserved_share"] = float(cue[rated].isin(["i", "igen"]).mean())
    else:
        result["cue_preserved_share"] = None
    result["cue_rated"] = int(rated.sum())

    # az első pár megjegyzés
    notes = [n for n in sample["notes"].dropna().astype(str) if n.strip() != ""]
    result["notes"] = notes[:8]
    return result


def plot_results(res, keys):
    # 1. ábra: pontosság és makro-F1 modellenként, a foldok közti szórással
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    x = np.arange(len(keys))
    width = 0.38

    for j, metric in enumerate(["accuracy", "macro_f1"]):
        means = [res["models"][k]["mean"][metric] for k in keys]
        stds = [res["models"][k]["std"][metric] for k in keys]
        if metric == "accuracy":
            hatch, label = "", "Pontosság"
        else:
            hatch, label = "//", "Makro-F1"

        bars = ax.bar(x + (j - 0.5) * width, means, width, yerr=stds, capsize=3, hatch=hatch,
                      color=[COLORS[k] for k in keys], edgecolor="#333", linewidth=0.6, label=label)

        # érték kiírása az oszlopok fölé (magyar tizedesvesszővel)
        for bar, mean, std in zip(bars, means, stds):
            text = f"{mean:.3f}".replace(".", ",")
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.008, text,
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x, [res["models"][k]["display"] for k in keys], fontsize=8)

    # az y tengely alja: a legkisebb (átlag - szórás) alatt kicsivel
    lowest = 1.0
    for k in keys:
        for metric in ["accuracy", "macro_f1"]:
            value = res["models"][k]["mean"][metric] - res["models"][k]["std"][metric]
            lowest = min(lowest, value)
    ax.set_ylim(max(0.0, np.floor((lowest - 0.05) * 10) / 10), 1.0)

    ax.set_ylabel("Érték (5-szörös CV átlag ± szórás)")
    ax.axhline(0.5, color="#999", lw=0.8, ls=":")  # véletlen tippelés szintje
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_results.png")
    plt.close(fig)


def plot_confusion(res, oof, keys):
    # 2. ábra: konfúziós mátrixok egymás mellett
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3), dpi=200)
    axes = np.atleast_1d(axes)  # ha csak 1 modell van, akkor is lista legyen

    for ax, k in zip(axes, keys):
        cm = confusion_matrix(oof[k]["label"], oof[k]["pred"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=800)
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > 400 else "black"  # sötét háttéren fehér szám
                ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
        ax.set_xticks([0, 1], ["valódi", "megtévesztő"], fontsize=8)
        ax.set_yticks([0, 1], ["valódi", "megtévesztő"], fontsize=8)
        ax.set_xlabel("Predikció", fontsize=8)
        ax.set_ylabel("Valós címke", fontsize=8)
        ax.set_title(res["models"][k]["display"], fontsize=9)

    fig.tight_layout()
    fig.savefig(RESULTS / "fig_confusion.png")
    plt.close(fig)


def main():
    res = {"models": {}, "comparisons": {}, "by_polarity": {}}
    oof = {}  # modellenként az out-of-fold predikciók

    # eredmények beolvasása (ami még nem futott le, azt kihagyjuk)
    for key, display_name in MODELS:
        metrics_path = RESULTS / f"metrics_{key}.json"
        if not metrics_path.exists():
            print(f"(nincs még eredmény: {key})")
            continue
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        m["display"] = display_name
        res["models"][key] = m
        df = pd.read_csv(RESULTS / f"oof_{key}.csv")
        oof[key] = df.sort_values("id").reset_index(drop=True)

    # külön a pozitív és a negatív véleményekre
    for key, df in oof.items():
        res["by_polarity"][key] = {}
        for polarity, group in df.groupby("polarity"):
            res["by_polarity"][key][polarity] = metrics(group["label"], group["pred"])

    # McNemar-próbák: a neurális modellek vs. magyar baseline, és hibrid vs. huBERT
    pairs = [("hubert", "tfidf_lr_hu"), ("hybrid", "tfidf_lr_hu"), ("hybrid", "hubert")]
    for a, b in pairs:
        if a in oof and b in oof:
            y = oof[a]["label"].to_numpy()
            res["comparisons"][f"{a}_vs_{b}"] = mcnemar(y, oof[a]["pred"].to_numpy(), oof[b]["pred"].to_numpy())

    res["mt_quality"] = mt_quality()

    # a baseline legfontosabb szavai
    for name in ["top_features_hu", "top_features_en"]:
        path = RESULTS / f"{name}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                res[name] = json.load(f)

    with open(RESULTS / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # ábrák
    keys = [k for k, _ in MODELS if k in res["models"]]
    if len(keys) > 0:
        plot_results(res, keys)

    cm_keys = [k for k in ["tfidf_lr_hu", "hubert", "hybrid"] if k in oof]
    if len(cm_keys) > 0:
        plot_confusion(res, oof, cm_keys)

    # rövid összefoglaló a konzolra
    print(json.dumps({k: v["mean"] for k, v in res["models"].items()}, indent=2))
    print(json.dumps(res["comparisons"], indent=2))


if __name__ == "__main__":
    main()
