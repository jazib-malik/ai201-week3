# TakeMeter — Local Interface (stretch feature)

A tiny app that loads your fine-tuned model and classifies a new r/nba comment with its
discourse label and confidence. Covers the brief's optional **Deployed interface** stretch feature.

## 1. Get the trained model out of Colab

After Section 3 finishes training, add a cell and run:

```python
trainer.save_model("takemeter-model")
import shutil
shutil.make_archive("takemeter-model", "zip", "takemeter-model")
from google.colab import files
files.download("takemeter-model.zip")
```

Unzip `takemeter-model.zip` into the **repo root** so the path is `./takemeter-model/`.
(It's gitignored — model weights are large and regenerable.)

## 2. Install deps

```bash
pip install transformers torch gradio
```

## 3. Run

```bash
python app/app.py          # Gradio web UI at http://127.0.0.1:7860
python app/app.py --cli    # simple terminal loop, no extra UI deps beyond transformers/torch
```

Point at a model elsewhere with `--model PATH` or the `TAKEMETER_MODEL` env var.

## What it shows

For any comment you paste: the predicted label (analysis / hot_take / reaction / banter), the
model's confidence, and the full probability distribution across all four labels — handy for the
demo video's "show 3–5 classifications with confidence" requirement.
