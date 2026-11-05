"""Gépi fordítás angolról magyarra, plusz egy minta a fordítás kézi ellenőrzéséhez.
A véleményeket először mondatokra bontjuk, és mondatonként fordítunk.
Erre azért van szükség, mert a fordító modell csak korlátozott hosszú
szöveget tud kezelni, így egy egész véleménynél levágná a végét.
"""

import argparse
import json
import re
import time

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .common import DATA_PROC, RESULTS, SEED, load_raw

MODELS = {
    "opus": "Helsinki-NLP/opus-mt-en-hu",
    "nllb": "facebook/nllb-200-distilled-600M",
}

# Mondathatár: . ! ? után szóköz, és utána nagybetű / szám / idézőjel / zárójel jön
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text, max_chars=400):
    # Szöveg feldarabolása mondatokra.
    # Van az adatban pár nagyon hosszú, írásjel nélküli rész, ezeket vesszőnél (ha nincs, akkor szóköznél) vágjuk kb. max_chars hosszúra.
    text = re.sub(r"\s+", " ", text).strip()
    result = []
    for sent in SENTENCE_END.split(text):
        while len(sent) > max_chars:
            cut = sent.rfind(", ", 0, max_chars)
            if cut < max_chars // 3:
                # nincs vessző elég hátul -> utolsó szóköznél vágunk
                cut = sent.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            result.append(sent[:cut + 1].strip())
            sent = sent[cut + 1:].strip()
        if sent:
            result.append(sent)
    return result


def translate_sentences(sentences, engine, batch_size):
    model_name = MODELS[engine]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # az NLLB-nek meg kell mondani a forrás- és a célnyelvet
    if engine == "nllb":
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    if device == "cuda":
        model = model.half()  # fp16: kevesebb memória, gyorsabb

    gen_settings = {"num_beams": 4, "max_new_tokens": 256}
    if engine == "nllb":
        gen_settings["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids("hun_Latn")

    # Hossz szerint rendezzük a mondatokat, így egy batch-ben hasonló hosszúak lesznek -> kevesebb padding, gyorsabb fordítás. A végén visszarakjuk őket az eredeti helyükre.
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]), reverse=True)
    translated = [""] * len(sentences)

    start_time = time.time()
    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]
        batch = [sentences[i] for i in batch_idx]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_settings)
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, t in zip(batch_idx, texts):
            translated[i] = t.strip()

        # időnként kiírjuk, hol tartunk
        if (start // batch_size) % 50 == 0:
            print(f"  {start + len(batch_idx)}/{len(order)} mondat kész, {time.time() - start_time:.0f} s")

    return translated


def word_count(series):
    # szavak száma minden sorban
    return series.str.split().str.len()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=list(MODELS), default="opus")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="csak az első N vélemény (teszteléshez)")
    parser.add_argument("--sample-size", type=int, default=30, help="hány véleményt ellenőrzünk kézzel")
    args = parser.parse_args()

    df = load_raw()
    if args.limit:
        df = df.head(args.limit).copy()

    # 1) mondatokra bontás; megjegyezzük, melyik mondat melyik véleményhez tartozik
    sentences = []
    review_of_sentence = []
    for review_id, text in zip(df["id"], df["text_en"]):
        for s in split_sentences(text):
            sentences.append(s)
            review_of_sentence.append(review_id)
    print(f"{len(df)} vélemény, {len(sentences)} mondat fordítása ({MODELS[args.engine]})")

    # 2) fordítás
    start_time = time.time()
    hu_sentences = translate_sentences(sentences, args.engine, args.batch_size)
    elapsed = time.time() - start_time

    # 3) a lefordított mondatok összerakása véleményekké
    hu_by_review = {}
    for review_id, t in zip(review_of_sentence, hu_sentences):
        if review_id not in hu_by_review:
            hu_by_review[review_id] = []
        hu_by_review[review_id].append(t)
    df["text_hu"] = [" ".join(hu_by_review.get(r, [])) for r in df["id"]]
    df["mt_engine"] = MODELS[args.engine]

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROC / "reviews_hu.csv", index=False)
    print(f"Mentve: {DATA_PROC / 'reviews_hu.csv'}")

    # 4) pár statisztika a fordításról
    RESULTS.mkdir(parents=True, exist_ok=True)
    info = {
        "engine": MODELS[args.engine],
        "n_reviews": int(len(df)),
        "n_sentences": len(sentences),
        "seconds": round(elapsed, 1),
        "words_en_mean": float(word_count(df["text_en"]).mean()),
        "words_hu_mean": float(word_count(df["text_hu"]).mean()),
        "empty_translations": int((df["text_hu"].str.len() == 0).sum()),
    }
    with open(RESULTS / "mt_meta.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(info)

    # 5) minta a kézi ellenőrzéshez: a 4 csoportból (valódi/megtévesztő x pozitív/negatív) ugyanannyit választunk
    per_group = max(1, args.sample_size // 4)
    parts = []
    for _, group in df.groupby(["deceptive", "polarity"]):
        parts.append(group.sample(min(per_group, len(group)), random_state=SEED))
    sample = pd.concat(parts)[["id", "deceptive", "polarity", "text_en", "text_hu"]]

    # üres oszlopok, ezeket kell majd kézzel kitölteni
    sample["adequacy_1_5"] = ""       # mennyire adja vissza a jelentést (1-5)
    sample["fluency_1_5"] = ""        # mennyire természetes magyar szöveg (1-5)
    sample["cue_preserved_i_n"] = ""  # megmaradtak-e a megtévesztésre utaló jelek (pl. nevek, részletek, hangnem)
    sample["notes"] = ""
    sample.to_csv(DATA_PROC / "mt_quality_sample.csv", index=False)
    print(f"Kézi ellenőrzéshez: {DATA_PROC / 'mt_quality_sample.csv'} ({len(sample)} vélemény)")


if __name__ == "__main__":
    main()
