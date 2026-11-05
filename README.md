# Megtévesztő szállodai vélemények felismerése magyar nyelven

Féléves feladat – **Molnár Barnabás**
Kapcsolódó diplomamunka: *Megtévesztő vélemények detektálása magyar nyelven transformer-alapú hibrid CNN–BiLSTM modellel* (konzulens: Sipos Miklós)

A feladat: magyar nyelvű szállodai véleményekről eldönteni, hogy **valódiak** vagy **megtévesztők** (kitaláltak). Magyar adathalmaz nincs ilyenre, ezért az angol Ott-féle Deceptive Opinion Spam Corpust (1600 vélemény, 20 chicagói szálloda, 4 × 400 vélemény) fordítjuk le gépi fordítással.

| Modell | Röviden |
|---|---|
| TF-IDF + LR | baseline: szavak és szópárok (1–2-gramok) + logisztikus regresszió |
| huBERT | a `SZTAKI-HLT/hubert-base-cc` modell finomhangolva osztályozásra |
| huBERT + CNN–BiLSTM | a huBERT token-kimenetei → Conv1D (k = 3, 5) → BiLSTM → max + átlag pooling |

Kiértékelés: **5-szörös keresztvalidáció, szállodák szerint csoportosítva** (egy szálloda összes véleménye ugyanabba a foldba kerül). Pontosságot és makro-F1-et számolunk, a modelleket páronként McNemar-próbával hasonlítjuk össze.

## Futtatás

**Colabban (ajánlott, T4 GPU-val):** meg kell nyitni a `notebooks/hu_deceptive_reviews.ipynb` notebookot, és végigfuttatni. A repóban csak a kód és az eredeti adat van, a fordítás, a foldok és az eredmények a notebookban készülnek el.

**Saját gépen:**
```bash
pip install -r requirements.txt   # + PyTorch külön
python -m src.translate --engine opus
python -m src.baseline --lang hu
python -m src.baseline --lang en
python -m src.neural --model hubert
python -m src.neural --model hybrid
python -m src.analyze
```

## Mappák

```
data/raw/deceptive-opinion.csv        az eredeti adat (Ott et al. 2011, 2013)
src/                                  a kód: fordítás, modellek, kiértékelés
data/processed/   (futtatáskor jön létre) magyar fordítás, foldok, kézi ellenőrzéshez a minta
results/          (futtatáskor jön létre) predikciók, metrikák, ábrák
notebooks/hu_deceptive_reviews.ipynb  az egész folyamat Colabban
```

## Adat és licenc

Deceptive Opinion Spam Corpus v1.4 – Myle Ott, Claire Cardie, Jeffrey T. Hancock, Yejin Choi; Creative Commons Attribution-NonCommercial-ShareAlike 3.0. Csak kutatási/oktatási célra használható.

- Ott, M., Choi, Y., Cardie, C., Hancock, J. T. (2011). Finding Deceptive Opinion Spam by Any Stretch of the Imagination. ACL-HLT.
- Ott, M., Cardie, C., Hancock, J. T. (2013). Negative Deceptive Opinion Spam. NAACL-HLT.
