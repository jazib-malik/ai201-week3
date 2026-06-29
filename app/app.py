"""
TakeMeter — local classify-a-comment interface (stretch feature).

Loads your fine-tuned DistilBERT model and labels a new r/nba comment with its
discourse type and a confidence score.

Setup:
    1. In Colab after training, save the model:  trainer.save_model("takemeter-model")
       then zip and download the `takemeter-model/` folder.
    2. Unzip it into this repo root (so the path is ./takemeter-model/).
    3. pip install transformers torch gradio
    4. Run:
         python app/app.py            # launches the Gradio web UI
         python app/app.py --cli      # simple terminal loop instead

The model directory already stores id2label/label2id (the notebook sets them when
it loads DistilBERT), so labels come back as strings automatically.
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.environ.get("TAKEMETER_MODEL", "takemeter-model")


def load_model(model_dir: str):
    if not os.path.isdir(model_dir):
        sys.exit(
            f"Model directory '{model_dir}' not found.\n"
            "Download your fine-tuned model from Colab (trainer.save_model(...)) and\n"
            "unzip it here, or set TAKEMETER_MODEL to its path."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def classify(text: str, tokenizer, model):
    """Return (label, confidence, full_distribution_dict)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(probs.argmax())
    label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id])
    dist = {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}
    return label, confidence, dist


def run_cli(tokenizer, model):
    print("TakeMeter — type an r/nba comment (blank line to quit).\n")
    while True:
        try:
            text = input("comment> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        label, conf, dist = classify(text, tokenizer, model)
        print(f"  → {label}  (confidence {conf:.2f})")
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        print("    " + "  ".join(f"{k}:{v:.2f}" for k, v in ranked) + "\n")


def run_gradio(tokenizer, model):
    import gradio as gr

    def predict(text):
        if not text.strip():
            return "—", {}
        label, conf, dist = classify(text, tokenizer, model)
        return f"{label}  ({conf:.0%} confident)", dist

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=4, label="r/nba comment", placeholder="Paste a comment..."),
        outputs=[gr.Label(label="Prediction"), gr.Label(label="Full distribution", num_top_classes=4)],
        title="TakeMeter",
        description="Classifies an r/nba comment as analysis / hot_take / reaction / banter.",
        examples=[
            ["When Draymond sits, GSW's defensive rating jumps from 108 to 117 and they get killed on the glass."],
            ["Tatum is top-3 in the league and it's not even close."],
            ["WHAT A SHOT. I'm shaking right now."],
            ["Knicks fans woke up and chose delusion today"],
        ],
    )
    demo.launch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run a terminal loop instead of the web UI.")
    parser.add_argument("--model", default=MODEL_DIR, help="Path to the fine-tuned model directory.")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    if args.cli:
        run_cli(tokenizer, model)
    else:
        run_gradio(tokenizer, model)


if __name__ == "__main__":
    main()
