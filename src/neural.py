"""huBERT finomhangolás és a huBERT + CNN-BiLSTM hibrid modell tanítása.

    python -m src.neural --model hubert
    python -m src.neural --model hybrid

Mindkét modell alapja ugyanaz az előtanított huBERT (SZTAKI-HLT/hubert-base-cc):
  * hubert: a [CLS] token kimenetére egy lineáris réteg kerül (sima finomhangolás)
  * hybrid: a huBERT összes token-kimenete megy tovább egy CNN-be (3-as és 5-ös
            kernel), utána egy BiLSTM-be, aztán max- és átlag-pooling, és a végén
            egy lineáris réteg
A huBERT is tanul a fejjel együtt, de a fej nagyobb tanulási rátát kap.
Nincs early stopping, a teszt foldot csak a tanítás végén nézzük meg.
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from transformers.modeling_outputs import SequenceClassifierOutput

from .common import RESULTS, load_hu, make_folds, metrics, save_oof, set_seed

ENCODER = "SZTAKI-HLT/hubert-base-cc"


class ReviewDataset(Dataset):
    # A tokenizált véleményeket és a címkéket adja vissza egyesével a DataLoadernek

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {}
        for key, values in self.encodings.items():
            item[key] = torch.tensor(values[i])
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


class HybridCNNBiLSTM(nn.Module):
    def __init__(self, encoder_name, n_filters=128, kernel_sizes=(3, 5),
                 lstm_hidden=128, dropout=0.3, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size  # huBERT-nél 768

        # két konvolúció párhuzamosan, 3-as és 5-ös ablakkal
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(nn.Conv1d(hidden, n_filters, kernel_size=k, padding=k // 2))

        self.lstm = nn.LSTM(n_filters * len(kernel_sizes), lstm_hidden,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # BiLSTM kimenet 2 * lstm_hidden, ebből max és átlag pooling -> 4 * lstm_hidden
        self.classifier = nn.Linear(4 * lstm_hidden, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        # (a token_type_ids-t a tokenizer adja, de itt nem használjuk)

        # 1) huBERT: minden tokenre egy 768 hosszú vektor
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # alak: (batch, tokenek, 768)

        # 2) CNN: a Conv1d a csatornákat a 2. dimenzióban várja, ezért transzponálunk
        x = self.dropout(hidden).transpose(1, 2)        # (batch, 768, tokenek)
        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)              # (batch, 256, tokenek)
        x = x.transpose(1, 2)                           # (batch, tokenek, 256)

        # 3) BiLSTM: a "pack" miatt a padding tokeneket nem dolgozza fel
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x.float(), lengths, batch_first=True,
                                                   enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True,
                                                       total_length=x.size(1))

        # 4) pooling: max és átlag, de csak a valódi tokeneken (paddinget kihagyva)
        mask = attention_mask.unsqueeze(-1).to(lstm_out.dtype)
        max_pool = lstm_out.masked_fill(mask == 0, -1e4).max(dim=1).values
        mean_pool = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # 5) osztályozás
        features = torch.cat([max_pool, mean_pool], dim=1)
        logits = self.classifier(self.dropout(features))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        # ugyanolyan formában adjuk vissza, mint a huBERT modell, így a tanító ciklusnak mindegy
        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_model(model_type):
    if model_type == "hubert":
        return AutoModelForSequenceClassification.from_pretrained(ENCODER, num_labels=2)
    else:
        return HybridCNNBiLSTM(ENCODER)


def make_param_groups(model, model_type, lr, head_lr, weight_decay):
    # A huBERT (kódoló) és az új rétegek (fej) külön tanulási rátát kapnak.
    # A bias és LayerNorm súlyokra nem teszünk weight decay-t (BERT-nél ez a szokás).
    no_decay_words = ["bias", "LayerNorm.weight", "layer_norm"]

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if model_type == "hybrid":
            is_head = not name.startswith("encoder.")
        else:
            is_head = name.startswith("classifier.")

        if is_head:
            head_params.append((name, param))
        else:
            encoder_params.append((name, param))

    groups = []
    for params, rate in [(encoder_params, lr), (head_params, head_lr)]:
        with_decay = []
        without_decay = []
        for name, param in params:
            if any(word in name for word in no_decay_words):
                without_decay.append(param)
            else:
                with_decay.append(param)
        groups.append({"params": with_decay, "lr": rate, "weight_decay": weight_decay})
        groups.append({"params": without_decay, "lr": rate, "weight_decay": 0.0})
    return groups


@torch.no_grad()
def predict(model, loader, device):
    # Visszaadja minden véleményre annak a valószínűségét, hogy megtévesztő
    model.eval()
    all_probs = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            out = model(**inputs)
        probs = torch.softmax(out.logits.float(), dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)


def select_rows(encodings, idx):
    # a tokenizált adatból csak az idx-ben lévő sorokat tartja meg
    result = {}
    for key, values in encodings.items():
        result[key] = [values[i] for i in idx]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hubert", "hybrid"], required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=None, help="alapból: hubert 2e-5, hybrid 1e-3")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--folds", type=str, default=None, help="pl. '0,1' - csak ezeket a foldokat futtatja (teszteléshez)")
    args = parser.parse_args()

    # a fej tanulási rátája: a hibridnél nagyobb, mert ott a CNN és LSTM nulláról tanul
    if args.head_lr is not None:
        head_lr = args.head_lr
    elif args.model == "hybrid":
        head_lr = 1e-3
    else:
        head_lr = args.lr

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_hu()
    folds = make_folds(df)
    labels = df["label"].to_numpy()

    # tokenizálás egyszer, az egész adatra
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encodings = tokenizer(df["text_hu"].tolist(), truncation=True, max_length=args.max_len)
    lengths = np.array([len(x) for x in encodings["input_ids"]])
    truncated = (lengths >= args.max_len).mean()
    print(f"Tokenhossz: medián {np.median(lengths):.0f}, 95% {np.percentile(lengths, 95):.0f}, "
          f"levágva: {truncated:.1%}")

    def collate(batch):
        # egy batch-en belül azonos hosszúra paddingeli a szövegeket
        return tokenizer.pad(batch, return_tensors="pt")

    if args.folds is None:
        fold_list = list(range(folds.max() + 1))
    else:
        fold_list = [int(x) for x in args.folds.split(",")]

    prob = np.full(len(df), np.nan)
    train_log = []

    for k in fold_list:
        set_seed()  # minden fold ugyanonnan induljon
        fold_start = time.time()
        train_idx = np.where(folds != k)[0]
        test_idx = np.where(folds == k)[0]

        train_loader = DataLoader(ReviewDataset(select_rows(encodings, train_idx), labels[train_idx]),
                                  batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        test_loader = DataLoader(ReviewDataset(select_rows(encodings, test_idx), labels[test_idx]),
                                 batch_size=args.batch_size * 2, collate_fn=collate)

        # minden foldnál új, friss modell
        model = build_model(args.model).to(device)
        optimizer = torch.optim.AdamW(make_param_groups(model, args.model, args.lr, head_lr, args.weight_decay))
        total_steps = len(train_loader) * args.epochs
        # tanulási ráta: az első 10%-ban felfut (warmup), utána lineárisan csökken
        scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
        # fp16 tanításhoz kell (GPU-n), CPU-n nem csinál semmit
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        for epoch in range(args.epochs):
            model.train()
            loss_sum = 0.0
            for batch in train_loader:
                batch = {key: v.to(device) for key, v in batch.items()}
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                    out = model(**batch)
                    loss = out.loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradiens vágás
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss_sum += loss.item()

            avg_loss = loss_sum / len(train_loader)
            print(f"  fold {k} | epoch {epoch + 1}/{args.epochs} | train loss {avg_loss:.4f}")
            train_log.append({"fold": k, "epoch": epoch + 1, "train_loss": avg_loss})

        # kiértékelés a teszt foldon
        prob[test_idx] = predict(model, test_loader, device)
        preds = (prob[test_idx] >= 0.5).astype(int)
        m = metrics(labels[test_idx], preds)
        print(f"fold {k}: pontosság {m['accuracy']:.4f}, makro-F1 {m['macro_f1']:.4f} "
              f"({time.time() - fold_start:.0f} s)")

        # memória felszabadítása a következő fold előtt
        del model, optimizer
        torch.cuda.empty_cache()

    if np.isnan(prob).any():
        print("Nem futott le minden fold, ezért nem mentjük az eredményeket.")
        return

    config = dict(vars(args))
    config["head_lr"] = head_lr
    config["encoder"] = ENCODER
    config["device"] = device
    config["token_len_median"] = float(np.median(lengths))
    config["truncated_share"] = float(truncated)
    save_oof(args.model, df, folds, prob, extra=config)

    with open(RESULTS / f"trainlog_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)


if __name__ == "__main__":
    main()
