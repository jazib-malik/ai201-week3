"""
TakeMeter end-to-end pipeline (local mirror of the Colab notebook).

Reproduces the notebook's setup so the numbers are representative of what the
notebook produces:
  - distilbert-base-uncased, 70/15/15 stratified split (seed 42)
  - 3 epochs, lr 2e-5, batch 16, weight decay 0.01, 50 warmup steps
  - same locked test set used for BOTH the fine-tuned model and the Groq baseline

Outputs (in outputs/):
  evaluation_results.json, confusion_matrix.png, finetuned_report.txt,
  baseline_report.txt, test_predictions.csv
Also saves the fine-tuned model to takemeter-model/ (for app/app.py).
"""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {"analysis": 0, "hot_take": 1, "reaction": 2, "banter": 3}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABELS = [ID2LABEL[i] for i in range(4)]

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------- data ----------
def load_split():
    rows = list(csv.DictReader(open(ROOT / "data" / "takemeter_nba.csv", encoding="utf-8")))
    texts = [r["text"] for r in rows]
    y = [LABEL_MAP[r["label"]] for r in rows]
    idx = list(range(len(rows)))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=[y[i] for i in tmp])
    pack = lambda ids: ([texts[i] for i in ids], [y[i] for i in ids])
    return pack(tr), pack(va), pack(te)


def to_loader(tokenizer, texts, labels, batch, shuffle):
    enc = tokenizer(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for input_ids, attn, labels in loader:
        out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
        logits_all.append(out.logits.cpu())
        labels_all.append(labels)
    logits = torch.cat(logits_all)
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(1)
    return preds, probs, torch.cat(labels_all).numpy()


# ---------- fine-tune ----------
def finetune(train, val, test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4, id2label=ID2LABEL, label2id=LABEL_MAP).to(device)

    tr_loader = to_loader(tok, *train, batch=16, shuffle=True)
    va_loader = to_loader(tok, *val, batch=32, shuffle=False)
    te_loader = to_loader(tok, *test, batch=32, shuffle=False)

    epochs = int(os.environ.get("TM_EPOCHS", "3"))
    lr = float(os.environ.get("TM_LR", "2e-5"))
    print(f"  hyperparams: epochs={epochs} lr={lr}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(tr_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=50, num_training_steps=total_steps)

    best_acc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for input_ids, attn, labels in tr_loader:
            opt.zero_grad()
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                        labels=labels.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        vp, _, vy = predict(model, va_loader, device)
        vacc = accuracy_score(vy, vp)
        print(f"  epoch {ep}: val accuracy {vacc:.3f}")
        if vacc >= best_acc:
            best_acc, best_state = vacc, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    tok.save_pretrained(ROOT / "takemeter-model")
    model.save_pretrained(ROOT / "takemeter-model")
    preds, probs, y = predict(model, te_loader, device)
    return preds, probs, y, tok


# ---------- baseline ----------
SYSTEM_PROMPT = """
You are classifying comments from the r/nba subreddit by the KIND of discourse each one is.
Assign each comment to exactly one of four categories.

analysis: Advances a basketball claim backed by specific, verifiable evidence — stats, game-film
detail, lineup/scheme observation, or a concrete historical comparison. If you removed the opinion,
the evidence would still stand on its own.
Example: "When Draymond sits, GSW's defensive rating jumps from 108 to 117 and they get killed on the glass — that's why that lineup works."

hot_take: A bold, confident opinion or prediction asserted without real supporting evidence. It may
drop a stat decoratively, but the claim rests on conviction, not reasoning.
Example: "Tatum is top-3 in the league and it's not even close. Anyone who disagrees doesn't watch basketball."

reaction: An immediate emotional response to a specific moment — a play, a trade, a final score, a
stat line. It expresses a feeling in the moment and makes no argument.
Example: "WHAT A SHOT. I'm shaking. Most clutch thing I've ever seen."

banter: Humor, memes, trash talk, or one-liners where the point is comedic or social rather than to
make a sincere basketball claim.
Example: "Knicks fans woke up and chose delusion today"

Respond with ONLY the label name in lowercase (analysis, hot_take, reaction, or banter).
Do not explain your reasoning. Do not add punctuation.

Valid labels:
analysis
hot_take
reaction
banter
"""


def parse_label(raw):
    raw = raw.strip().lower()
    for label in sorted(LABEL_MAP, key=len, reverse=True):
        if raw == label or label in raw:
            return label
    return None


def baseline(test_texts):
    from dotenv import load_dotenv
    from groq import Groq
    load_dotenv(ROOT / ".env")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    preds = []
    for i, t in enumerate(test_texts):
        try:
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": f"Classify this post:\n\n{t}"}],
                temperature=0, max_tokens=20)
            preds.append(parse_label(r.choices[0].message.content))
        except Exception as e:
            print("  groq error:", e)
            preds.append(None)
        time.sleep(0.1)
    return preds


# ---------- main ----------
def main():
    train, val, test = load_split()
    print(f"Split: train {len(train[0])} / val {len(val[0])} / test {len(test[0])}")

    print("\n[1/2] Zero-shot baseline (Groq llama-3.3-70b)...")
    bl_raw = baseline(test[0])
    valid = [(parse, true) for parse, true in zip(bl_raw, test[1]) if parse is not None]
    bl_pred = [LABEL_MAP[p] for p, _ in valid]
    bl_true = [t for _, t in valid]
    bl_acc = accuracy_score(bl_true, bl_pred)
    bl_macro_f1 = f1_score(bl_true, bl_pred, average="macro")
    bl_report = classification_report(bl_true, bl_pred, target_names=LABELS, zero_division=0)
    unparseable = bl_raw.count(None)
    print(f"  baseline accuracy {bl_acc:.3f} | macro-F1 {bl_macro_f1:.3f} | unparseable {unparseable}/{len(test[0])}")

    print("\n[2/2] Fine-tuning DistilBERT...")
    ft_pred, ft_probs, ft_true, tok = finetune(train, val, test)
    ft_acc = accuracy_score(ft_true, ft_pred)
    ft_macro_f1 = f1_score(ft_true, ft_pred, average="macro")
    ft_report = classification_report(ft_true, ft_pred, target_names=LABELS, zero_division=0)
    print(f"  fine-tuned accuracy {ft_acc:.3f} | macro-F1 {ft_macro_f1:.3f}")

    # confusion matrix
    cm = confusion_matrix(ft_true, ft_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Fine-Tuned Model — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)

    # exports
    (OUT / "finetuned_report.txt").write_text(ft_report)
    (OUT / "baseline_report.txt").write_text(bl_report)
    json.dump({
        "baseline_accuracy": round(bl_acc, 4),
        "baseline_macro_f1": round(bl_macro_f1, 4),
        "finetuned_accuracy": round(ft_acc, 4),
        "finetuned_macro_f1": round(ft_macro_f1, 4),
        "improvement_accuracy": round(ft_acc - bl_acc, 4),
        "improvement_macro_f1": round(ft_macro_f1 - bl_macro_f1, 4),
        "test_set_size": len(test[0]),
        "baseline_unparseable": unparseable,
        "label_map": LABEL_MAP, "model": MODEL_NAME,
        "confusion_matrix": cm.tolist(), "confusion_labels": LABELS,
    }, open(OUT / "evaluation_results.json", "w"), indent=2)

    with open(OUT / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "true", "pred_finetuned", "confidence", "pred_baseline"])
        for i, t in enumerate(test[0]):
            w.writerow([t, ID2LABEL[test[1][i]], ID2LABEL[ft_pred[i]],
                        round(float(ft_probs[i][ft_pred[i]]), 3),
                        bl_raw[i] if bl_raw[i] else "UNPARSEABLE"])

    print("\n=== SUMMARY ===")
    print(f"Baseline  : acc {bl_acc:.3f}  macroF1 {bl_macro_f1:.3f}")
    print(f"Fine-tuned: acc {ft_acc:.3f}  macroF1 {ft_macro_f1:.3f}")
    print("\nFine-tuned per-class:\n" + ft_report)
    print("Outputs written to outputs/.")


if __name__ == "__main__":
    main()
