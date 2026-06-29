# Running TakeMeter on Google Colab — step by step

You have two ways to get results: the local `src/run_pipeline.py` (already produced the committed
`outputs/`), or the **Colab notebook** below — which is the course's official flow and gives you a GPU run
to show in your demo. The notebook **`takemeter_colab.ipynb`** in this folder is already filled in
(LABEL_MAP, the Groq prompt, and `num_train_epochs=10`), so you just upload and run.

**You'll need two files from this repo on your computer:**
- `notebook/takemeter_colab.ipynb`  (the filled notebook)
- `data/takemeter_nba.csv`  (the labeled dataset you upload when prompted)

---

## Steps

1. **Open Colab** → https://colab.research.google.com

2. **Upload the notebook.** `File → Upload notebook →` select `notebook/takemeter_colab.ipynb`.

3. **Set the GPU.** `Runtime → Change runtime type → T4 GPU → Save`.

4. **Add your Groq key as a secret.**
   - Click the **🔑 key icon** in the left sidebar.
   - `+ Add new secret` → Name: **`GROQ_API_KEY`** → Value: your Groq key.
   - Toggle **"Notebook access"** ON for it.
   *(Don't paste the key into a cell — the notebook reads it from this secret.)*

5. **Run the cells top to bottom** (`Runtime → Run all`, or Shift+Enter through them). What to watch for:
   - **Install + imports** → should print `GPU available: True`. If it says False, redo step 3.
   - **Section 1 (LABEL_MAP)** → prints the 4 labels. Already filled in — just run it.
   - **The upload cell** → a "Choose Files" button appears; pick **`data/takemeter_nba.csv`** from this repo.
   - **Validation cell** → should say `✅ All labels match` and show `analysis 50 / hot_take 50 /
     reaction 50 / banter 50`.
   - **Section 2** → confirms `Train: 140 / Validation: 30 / Test: 30`.
   - **Section 3 (fine-tune)** → trains for **10 epochs**, ~5–15 min on the T4. Watch validation accuracy
     climb across epochs.
   - **Section 4** → prints per-class metrics, saves `confusion_matrix.png`, lists wrong predictions.
   - **Section 5 (baseline)** → the prompt cell is already filled; this calls Groq once per test comment
     (~30 calls). If it flags unparseable responses >10%, your key/prompt has an issue — but the prompt is
     pre-tested, so it should be 0.
   - **Section 6** → prints the side-by-side comparison and writes `evaluation_results.json`.

6. **Download the outputs.** Open the **📁 Files** panel (left) → right-click each → Download:
   - `evaluation_results.json`
   - `confusion_matrix.png`
   Drop them into this repo's `outputs/` folder (you can overwrite the committed copies, or keep both).

7. **(Optional) Save the model for a live demo.** Add a cell at the end and run it:
   ```python
   trainer.save_model("takemeter-model")
   import shutil; shutil.make_archive("takemeter-model", "zip", "takemeter-model")
   from google.colab import files; files.download("takemeter-model.zip")
   ```
   Unzip into the repo root to use with `app/app.py`. *(You already have a locally-trained model in
   `takemeter-model/`, so this is only needed if you want the Colab-trained one.)*

---

## What to expect

Roughly: **fine-tuned accuracy ≈ 0.70, baseline ≈ 0.73** (macro-F1 ≈ 0.69 vs 0.72). Exact numbers vary a
little run to run (GPU nondeterminism + a 30-example test set), so don't worry if you're ±0.03 from the
committed results. The shape should hold: the two models roughly tie, the fine-tuned model is strongest on
`analysis`/`banter` and weakest on `reaction`. If your fine-tuned accuracy comes out near 0.50, you're
probably on 3 epochs — confirm Section 3 says `num_train_epochs=10`.

## If the runtime disconnects

Colab drops idle sessions. If that happens, re-run from the top: install/imports → Section 1 → re-upload
the CSV → Section 2 → then 3/4/5/6. You don't lose the notebook, just the in-memory state.
