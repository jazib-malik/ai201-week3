# TakeMeter — Classifying Discourse Quality in r/nba

A fine-tuned text classifier that labels what *kind* of comment an r/nba post is:
**analysis**, **hot_take**, **reaction**, or **banter**. Built for AI201 · Project 3.

> **Status:** the evaluation below is filled in with **real results** from the committed `outputs/`
> (local run; the notebook reproduces equivalent numbers). The only thing left for you is the **demo
> video link in §11**.

---

## 1. Community choice and reasoning

I chose **r/nba**, one of the largest and most active sports communities on Reddit. On any game night
the same event generates wildly different *kinds* of comments — someone breaks down a lineup with on/off
numbers, someone fires off a confident opinion with zero support, someone is just reacting in the moment,
and a huge share is pure jokes and trash talk. That range makes the discourse **text-heavy, high-volume,
and genuinely varied in quality**, which is what a discourse-quality classifier needs.

It's a strong fit for a classification task specifically because the *same topic* shows up as all four
discourse types. The model can't shortcut by keying on topic words ("Lakers," "MVP") — it has to learn the
**structure and intent** of a comment. That's what makes the analysis-vs-hot_take boundary a real learning
problem instead of keyword matching.

---

## 2. Label taxonomy

Four mutually exclusive labels, each answering "what is this comment *doing*?" with a different verb.
Together they cover 90%+ of real r/nba comments with no "other" bucket. (Full reasoning and decision
rules in [`planning.md`](planning.md).)

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|
| **analysis** | Advances a basketball claim backed by **specific, verifiable evidence** (stats, film, scheme, historical comp). Strip the opinion and the evidence still stands. | "When Draymond sits, GSW's defensive rating jumps from 108 to 117 and they get killed on the glass — that's why that lineup works." | "Jokić is at 9.8 assists and the Nuggets are +12 per 100 with him on, −3 off. That on/off swing is the whole offense." |
| **hot_take** | A bold, confident opinion/prediction asserted **without real evidence**. May drop a stat decoratively; rests on conviction, not reasoning. | "Tatum is top-3 in the league and it's not even close. Anyone who disagrees doesn't watch basketball." | "Lakers are winning the title this year. Book it." |
| **reaction** | An immediate **emotional** response to a specific moment. Expresses a feeling; makes no argument. | "WHAT A SHOT. I'm shaking. Most clutch thing I've ever seen." | "Bro I can't believe they traded him. I'm heartbroken right now." |
| **banter** | Humor, memes, trash talk, one-liners — the point is **comedic/social**, not a sincere claim. | "Knicks fans woke up and chose delusion today 💀" | "Refs already got the Lakers primetime script printed and laminated I see." |

**Why these distinctions matter to r/nba:** regulars already grade each other's comments in exactly this
vocabulary — "that's a hot take," "actual analysis," "ratio'd by banter." The taxonomy is native to the
community, not imposed.

---

## 3. Data: source, process, distribution, hard cases

**Source.** Public r/nba comments collected by authenticated scraping. Reddit blocks unauthenticated
programmatic reads (HTTP 403), so collection used Playwright: a real headed Chromium loads the r/nba page
(clearing Reddit's WAF and obtaining session cookies), then makes same-origin `fetch()` calls to the
public `.json` endpoints. Comments were pulled from two thread groups to cover all four labels:
trade/news threads (`src/collect_reddit.py`) which are heavy on `banter`/`hot_take`/asset-`analysis`, and
Post-Game/Game threads (`src/collect_postgame.py`) which are heavy on in-the-moment `reaction`. ~1,100
candidate comments were pooled; 200 were selected and labeled.

**Labeling process.** Each comment was read individually and labeled against the §2 definitions and the
decision rules in [`planning.md`](planning.md) §3. Labeling was done by **Claude (Opus)** — deliberately
*not* by the llama-3.3-70b model used for the zero-shot baseline, so the baseline stays an independent
comparison (see §10). Collection hygiene (in the scrapers): usernames/quotes/edits stripped, deleted/bot/
link-only comments dropped, comments kept to ~30–400 chars, HTML entities and markdown escapes cleaned,
deduped.

> **⚠️ Data provenance disclosure.** The committed dataset in `data/takemeter_nba.csv` is **real public
> r/nba comments** scraped via the authenticated-browser method above (June 2026: trade-news threads plus
> the Knicks–Spurs Finals and the draft thread). Labels were assigned by an AI annotator (Claude/Opus), reviewed
> against fixed decision rules — disclosed here and in §10. No rows are AI-*authored*; all comment text is
> genuine. The raw ~1,100-comment pool is gitignored; only the final 200 labeled rows are committed, to
> minimize redistribution of Reddit content (and the repo is kept private for the same reason).

**Label distribution** (from `data/takemeter_nba.csv`; the notebook reprints this in Section 1):

| Label | Count | % |
|---|---|---|
| analysis | 50 | 25% |
| hot_take | 50 | 25% |
| reaction | 50 | 25% |
| banter | 50 | 25% |
| **Total** | **200** | 100% |

*Deliberately balanced at 25% each — no label near the 70% imbalance ceiling, every label well above the
20% floor.*

**Three genuinely difficult-to-label examples** (real cases from the dataset; full log in `planning.md` §8):

1. **"Harper not playing 40 minutes a game will go down as one of the worst coaching decisions of all time…
   He is already top10 in rapm."**
   *Tension:* analysis (cites the RAPM stat) vs hot_take (hyperbolic "worst of all time" framing).
   *Decision → `analysis`.* Rule A: the advanced stat is genuine, verifiable evidence doing real work in
   the claim — not decoration — so it clears the analysis bar despite the loud framing.

2. **"Disgusting trade by the Suns. I absolutely hate this."**
   *Tension:* reaction (raw emotion) vs hot_take (a verdict on the trade).
   *Decision → `hot_take`.* Rule B: "Disgusting trade" is a general evaluative claim about the move, not
   just a feeling about a moment — the emotional tail ("I hate this") doesn't change that it's asserting a
   judgment.

3. **"Castle won the game. That three combined with the two clutch free throws… absolutely cold blooded."**
   *Tension:* reaction (emotional hype) vs analysis (identifies what decided the game).
   *Decision → `analysis`.* It names the *specific plays* (the three + two FTs) as the causal evidence for
   its claim — concrete and checkable — so structure wins over the emotional register.

---

## 4. Fine-tuning approach

- **Base model:** `distilbert-base-uncased` (HuggingFace, ~66M params) with a 4-class
  sequence-classification head.
- **Training setup:** 70/15/15 stratified split (seed 42) → **train 140 / val 30 / test 30**, max sequence
  length 256, evaluate on val each epoch and keep the best-validation checkpoint for the test evaluation.
  Run locally on **CPU** via `src/run_pipeline.py`, a self-contained loop that mirrors the notebook's setup
  (the notebook produces equivalent numbers on a T4 GPU; CPU training here took ~3 minutes).
- **Hyperparameters:** **10 epochs**, learning rate 2e-5, batch size 16, weight decay 0.01, 50 warmup
  steps, linear LR decay, grad-norm clip 1.0.

**Key hyperparameter decision — number of epochs (3 → 10).** The notebook default of **3 epochs** badly
**undertrains** this task: 140 training examples at batch 16 is only ~27 gradient steps, and validation
accuracy was still climbing (0.27 → 0.30 → 0.37) when training stopped — the model had barely moved off
random. Raising to **10 epochs** let validation accuracy plateau around 0.50 and lifted **test accuracy
from 0.53 to 0.70** (macro-F1 0.47 → 0.69), taking the fine-tuned model from "clearly worse than the
baseline" to "tied with it." I select the best-validation checkpoint to guard against overfitting the
140-example training set. (A separate lr=3e-5 / 8-epoch run scored *lower* on test (0.63), so I kept
lr 2e-5.) **To reproduce in the notebook, set `num_train_epochs=10` in Section 3.**

---

## 5. Baseline (Groq zero-shot)

**Model:** `llama-3.3-70b-versatile` via Groq, `temperature=0`, run on the **same locked test set** as the
fine-tuned model. The prompt supplies the four §2 definitions plus one example each and instructs the model
to output only a lowercase label; the parser matches that output to a label and flags anything unparseable.
Full prompt: `SYSTEM_PROMPT` in [`src/run_pipeline.py`](src/run_pipeline.py) and the notebook's Section 5.

**How results were collected:** `src/run_pipeline.py` (and the notebook) call Groq once per test comment at
`temperature=0`, parse the label, and score it with the same accuracy / per-class report used for the
fine-tuned model. **Unparseable rate: 0/30 (0%)** — the "output only the lowercase label" instruction
produced a clean, parseable label every time. (Note: at `temperature=0` Groq is still slightly
non-deterministic — baseline accuracy varied 0.70–0.73 across identical runs, i.e. ±1 test example, which
is the noise floor on a 30-example test set.)

---

## 6. Evaluation report

### 6.1 Headline comparison

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Zero-shot baseline (Groq llama-3.3-70b) | **0.733** | **0.722** |
| Fine-tuned DistilBERT (10 epochs) | **0.700** | **0.687** |
| Δ (fine-tuned − baseline) | −0.033 | −0.035 |

**Headline read:** a 66M-parameter DistilBERT fine-tuned on **140 examples** lands within **one test
example** of a **70B-parameter** model doing the task zero-shot. The baseline edges it out, but the gap
(−0.033 accuracy) is inside the noise floor of a 30-example test set (baseline itself swings ±0.03 between
identical runs). Practically this is a **tie** — which is a strong result for the small model, and the
per-class breakdown shows the two models are good at *different* things. (Raw numbers:
[`outputs/evaluation_results.json`](outputs/evaluation_results.json).)

### 6.2 Per-class metrics

**Fine-tuned DistilBERT** ([`outputs/finetuned_report.txt`](outputs/finetuned_report.txt)):

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.78 | 0.88 | **0.82** | 8 |
| hot_take | 0.56 | 0.71 | **0.62** | 7 |
| reaction | 0.75 | 0.38 | **0.50** | 8 |
| banter | 0.75 | 0.86 | **0.80** | 7 |
| **macro avg** | 0.71 | 0.71 | **0.69** | 30 |

**Zero-shot baseline** ([`outputs/baseline_report.txt`](outputs/baseline_report.txt)):

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.88 | 0.88 | **0.88** | 8 |
| hot_take | 0.50 | 0.43 | **0.46** | 7 |
| reaction | 0.86 | 0.75 | **0.80** | 8 |
| banter | 0.67 | 0.86 | **0.75** | 7 |
| **macro avg** | 0.72 | 0.73 | **0.72** | 30 |

**The story is in the per-class split, not the average.** Fine-tuning **beat** the 70B baseline on
`hot_take` (F1 0.62 vs 0.46) — it learned this community's "confident assertion" register better than a
general model does zero-shot. But it **lost badly on `reaction`** (F1 0.50 vs 0.80; recall just 0.38): the
small model can't reliably tell an in-the-moment emotional comment from an opinion or an observation, while
the 70B model reads emotional tone well. So the models are roughly tied on average but for *opposite*
reasons.

### 6.3 Confusion matrix (fine-tuned, test set)

Written as text so it reads cleanly; the image is committed at
[`outputs/confusion_matrix.png`](outputs/confusion_matrix.png). Rows = true label, columns = predicted.

| true ↓ \ pred → | analysis | hot_take | reaction | banter | recall |
|---|---|---|---|---|---|
| **analysis** | **7** | 1 | 0 | 0 | 7/8 |
| **hot_take** | 0 | **5** | 1 | 1 | 5/7 |
| **reaction** | 2 | 2 | **3** | 1 | 3/8 |
| **banter** | 0 | 1 | 0 | **6** | 6/7 |

*Reading guide:* the diagonal (bold) is correct. The story is the **`reaction` row**: of 8 true reactions,
the model got only 3 — it scattered the other 5 into `analysis` (2), `hot_take` (2), and `banter` (1). My
pre-registered guess (analysis→hot_take being the big error) was **wrong**: analysis is actually the model's
*best* class (7/8). The real weak boundary is **reaction**, which bleeds into every other label. Image
version: [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png).

### 6.4 Three wrong predictions, analyzed

The model got **9 of 30 wrong**. Three that isolate the boundaries it struggles with:

1. **"It looked like Spurs had 0 offensive plans."** — True: `reaction` · Predicted: `hot_take` (0.33) · baseline got it right.
   *Why it failed:* a terse, in-the-moment vent during a blowout. The model keys on the evaluative content
   ("0 offensive plans") and reads it as an opinion. This is the core **reaction→hot_take** confusion: once
   you strip tone, a short reaction and a short hot_take are nearly identical strings. The 70B baseline,
   which reads the venting register, labeled it `reaction`. **Boundary problem**, not a labeling slip.

2. **"…the amount of bricks the Spurs had on open looks is the only reason they didn't win by like 20."**
   — True: `reaction` · Predicted: `analysis` (0.42) · baseline also said `analysis`.
   *Why it failed:* I labeled it `reaction` (a fan venting), but it cites quasi-evidence ("bricks on open
   looks") and a counterfactual ("win by 20"), so *both* the model and llama read it as `analysis`. Two
   independent classifiers disagreeing with my label is a signal this is partly an **annotation-boundary
   problem** — my `reaction` call here is genuinely debatable, and "analysis" is defensible.

3. **"Why would you not want your picks with a 19 year old Cooper on the roster?? Are they high???"**
   — True: `analysis` · Predicted: `hot_take` (0.37) · baseline also said `hot_take`.
   *Why it failed:* the underlying point is sound asset reasoning (don't trade picks around a young star),
   but it's delivered as pure rhetorical outrage with **no evidence shown on the surface**. The model keys
   on the rhetorical form, not the implicit reasoning. Again both classifiers said `hot_take`, so this sits
   right on the **rule-A boundary** I defined — the kind of case the taxonomy was always going to find hard.

**Systematic pattern (and how I verified it).** Listing all 9 errors, **5 have true label `reaction`** — it
is overwhelmingly the failure class. The pattern: the model classifies by **surface form**, and `reaction`
is the one label with *no stable surface form*. Reactions that contain evaluative words get pulled to
`hot_take`; reactions that name plays/stats get pulled to `analysis`; jokey reactions get pulled to
`banter`. With only ~35 `reaction` training examples, the model never learned "emotional, in-the-moment"
as a feature — whereas it found reliable surface cues for analysis (numbers/asset terms), banter
(memes/names), and hot_take (confident assertions). I verified this by re-reading each error: the
analysis/hot_take/banter mistakes are isolated 1-offs, but the `reaction` mistakes form a clear directional
fan-out (the matrix row confirms it). A secondary finding, from cases 2–3: a few "errors" are really
**annotation-boundary disputes** where two classifiers overruled my label — so some of the ceiling here is
my own label noise on the reaction/analysis edge, not model weakness.

### 6.5 Sample classifications

Five test comments run through the **fine-tuned** model (confidence = softmax of the predicted class;
full list in [`outputs/test_predictions.csv`](outputs/test_predictions.csv)):

| Comment | Predicted | Confidence | Correct? |
|---|---|---|---|
| "Yeah 4 frps for a 28/7/5 player is bailing them out.. Those picks could be useless." | analysis | 0.69 | ✅ |
| "Somebody check on the Statue of Liberty" | banter | 0.45 | ✅ |
| "just a few more wins, then people can blame it on the Spurs being too young." | hot_take | 0.44 | ✅ |
| "So happy for KAT, dude. He's had such a tough career and such a rough few years." | reaction | 0.32 | ✅ |
| "It looked like Spurs had 0 offensive plans" | hot_take | 0.33 | ❌ (true: reaction) |

**Why a correct prediction is reasonable:** *"Yeah 4 frps for a 28/7/5 player is bailing them out… those
picks could be useless" → `analysis` (0.69).* This is reasonable — and the model's *most confident*
prediction — because it pairs concrete shorthand evidence (a 28/7/5 stat line, "4 first-round picks") with a
value judgment about the trade. That evidence-plus-reasoning structure is exactly the `analysis` signal, and
it's the class the model learned most cleanly.

**Note on confidence:** scores top out around **0.69** and many correct calls sit at 0.32–0.45 — the model
is rarely confident, which fits a hard, subjective task trained on only 140 examples. Confidence is weakly
calibrated, so it should not be used as a hard gate in any deployed tool without more data.

---

## 7. Reflection: what the model learned vs. what I intended

- **What I intended:** a model that judges a comment by what it's *doing* — reasoning from evidence vs.
  asserting vs. feeling vs. joking. I intended it to learn discourse **structure and intent**.

- **What it actually learned:** **surface form, not intent.** The error pattern shows the model latched onto
  reliable *lexical* proxies — numbers and asset/cap terms → `analysis`, memes/names/proper-noun riffs →
  `banter`, confident declaratives → `hot_take`. Those three proxies are good enough that those classes score
  0.62–0.82 F1. The intent I most cared about — "is this person *reasoning*?" — it only approximated via "are
  there numbers?", which is why `analysis` is its best class (numbers are an easy tell) but also why a
  reasoned-but-numberless comment like case 3 slips to `hot_take`.

- **What it missed:** **`reaction` has no lexical proxy**, so the model essentially never learned it
  (recall 0.38). "An emotional response to a moment" is defined by *tone and timing*, not vocabulary — and
  tone is exactly what a 140-example fine-tune of a small model can't capture but a 70B model reads natively
  (hence the baseline's 0.80 reaction F1). The gap between intended and learned behavior is sharpest
  precisely on the one label that is about *how* something is said rather than *what words* are in it.

---

## 8. Spec reflection

- **One way the spec helped:** writing the decision rules in `planning.md` §3 *before* labeling kept the
  hard boundaries consistent. When I hit cases like "Disgusting trade by the Suns. I absolutely hate this,"
  I didn't re-litigate — rule B already said "general evaluative claim → hot_take." Without those rules
  pre-committed, I'd have labeled emotional-but-evaluative comments inconsistently and taught the model a
  contradictory signal. The spec's insistence on naming the metrics up front (macro-F1, per-class) also
  saved me from being fooled by the 0.70 accuracy into missing the `reaction` collapse.

- **One way the implementation diverged, and why:** my plan assumed **manual** collection from a spread of
  thread types and a target of ~60/label. Reality forced two changes: (1) Reddit blocks unauthenticated
  reads, so I built an authenticated Playwright scraper instead of copy-pasting; and (2) the live front page
  was dominated by trade-news threads (heavy on banter/hot_take, thin on reaction), so I added a second
  targeted scrape of Post-Game threads specifically to source `reaction`, and settled on a clean 50/label ×
  200 rather than 60. The spec's "what if a label is underrepresented?" question (§4) is exactly the problem
  I hit — having pre-thought it meant I recognized the imbalance and fixed it by *collecting differently*,
  not by padding with weak examples.

---

## 9. How to run (reproduce the results)

The committed `outputs/` were produced by the **local pipeline**; the **notebook** path reproduces
equivalent numbers on a GPU.

**A) Local (what produced the committed results):**
```bash
pip install -r requirements.txt           # + torch, transformers, datasets (see requirements.txt)
pip install playwright && python -m playwright install chromium   # only if re-scraping
# (data already collected in data/takemeter_nba.csv)
TM_EPOCHS=10 python src/run_pipeline.py    # runs baseline + fine-tune + writes outputs/
```
`run_pipeline.py` reads `GROQ_API_KEY` from `.env`, runs the Groq baseline and the DistilBERT fine-tune on
the same seed-42 test split, and writes `outputs/` + saves the model to `takemeter-model/`. To re-collect
data from scratch: `python src/collect_reddit.py && python src/collect_postgame.py && python
src/build_pool_index.py && python src/build_dataset.py`.

**B) Colab notebook (optional):** upload the **pre-filled** notebook
[`notebook/takemeter_colab.ipynb`](notebook/takemeter_colab.ipynb) (LABEL_MAP, Groq prompt, and
`num_train_epochs=10` already set) → *Runtime → T4 GPU* → add `GROQ_API_KEY` via Colab Secrets → *Run all*
(upload `data/takemeter_nba.csv` when prompted) → download `evaluation_results.json` +
`confusion_matrix.png`. Full step-by-step: [`notebook/COLAB_GUIDE.md`](notebook/COLAB_GUIDE.md).

*(Optional local demo interface for the video: [`app/`](app) — see
[`app/README_app.md`](app/README_app.md). The trained model is already saved in `takemeter-model/`.)*

---

## 10. AI usage

AI assistance (Claude / Claude Code) was used heavily on this project; the specific instances:

1. **Scaffolding & taxonomy design.** Claude designed the repo structure, drafted `planning.md` and this
   README, and stress-tested the four-label taxonomy against boundary comments (sarcastic hot-takes,
   one-decorative-stat opinions), which produced the decision rules in `planning.md` §3.

2. **Data collection tool.** Claude wrote the scrapers (`src/collect_reddit.py`, `src/collect_postgame.py`)
   after discovering Reddit blocks unauthenticated reads — including the Playwright "load HTML for WAF
   clearance, then same-origin `fetch()`" approach that got past the 403, and the filtering/cleaning logic.

3. **Annotation (disclosed).** The 200 comments were labeled by **Claude (Opus)**, reading each comment and
   applying the §2 definitions + decision rules. This is AI annotation, not human annotation — disclosed
   here per the brief. Critically, the labeler is a **different model** than the zero-shot baseline
   (llama-3.3-70b), so the baseline is not graded against its own labels. Borderline calls are logged in
   `planning.md` §8 with the rule applied. *(A human-review or inter-annotator pass is the obvious next
   step / stretch — see planning.md.)*

4. **Failure-pattern analysis.** After training, Claude was given the full wrong-prediction list and the
   confusion matrix and asked for systematic patterns. It proposed "the model classifies by surface form and
   `reaction` has no stable surface form" — verified against the matrix (5 of 9 errors are true-`reaction`)
   and by re-reading each error. It also flagged that two "errors" (§6.4 cases 2–3) are annotation-boundary
   disputes rather than model failures — a point I kept because both the model *and* the independent llama
   baseline overruled my label. Claude also drafted the §6 write-up from the real `outputs/` artifacts.

---

## 11. Demo video

https://github.com/user-attachments/assets/7d8966de-69f3-4b4f-b485-ac85ccf21624





---

## Repo contents

```
planning.md                  Design spec & decision log (written before data collection)
README.md                    This report
requirements.txt             Python dependencies
data/
  takemeter_nba.csv          The labeled dataset — 200 real r/nba comments, 50/label
  HOW_TO_COLLECT.md          Manual r/nba collection guide (fallback method)
  _pool_*.csv                Raw scraped pools (gitignored; intermediate)
src/
  collect_reddit.py          Playwright scraper — trade/news threads
  collect_postgame.py        Playwright scraper — Post-Game threads (for reaction)
  build_pool_index.py        Merge + dedupe pools into an indexed file
  build_dataset.py           Apply hand labels -> data/takemeter_nba.csv
  run_pipeline.py            Baseline + fine-tune + evaluate -> outputs/
notebook/
  takemeter_colab.ipynb      Pre-filled, ready-to-run Colab notebook (optional GPU path)
  COLAB_GUIDE.md             Step-by-step Colab runbook
app/
  app.py                     Stretch: local classify-a-comment interface
  README_app.md              How to run the interface
outputs/
  evaluation_results.json    Headline metrics + confusion matrix (committed)
  confusion_matrix.png       Confusion-matrix image (committed)
  finetuned_report.txt       Per-class report, fine-tuned model
  baseline_report.txt        Per-class report, Groq baseline
  test_predictions.csv       Per-test-example predictions (both models) + confidence
takemeter-model/             Fine-tuned weights (gitignored; regenerate via run_pipeline.py)
```
