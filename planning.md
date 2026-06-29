# TakeMeter — Planning & Design Spec

**Project:** AI201 · Project 3 — Show What You Know: TakeMeter
**Community:** r/nba
**Task:** Fine-tune a text classifier that labels the *kind of discourse* an r/nba comment is.

This document is my working design log. It was written **before** I collected and
annotated any data, and updated as I made annotation decisions. The polished, reader-facing
write-up lives in `README.md`; this file holds the reasoning behind each decision.

---

## 1. Community

**What community and why.** I chose **r/nba**. It is one of the largest, most active sports
communities on Reddit, and on any given game night the same event produces wildly different
*kinds* of comments: a few people break down why a lineup worked using on/off numbers, a lot
of people fire off confident opinions with no support, others are just reacting in the moment
("I'M SHAKING"), and a huge share is pure jokes and trash talk. That range is exactly what makes
it a good classification target — the discourse is **text-heavy, high-volume, and genuinely varied
in quality**, and r/nba regulars already talk about each other's comments using this vocabulary
("that's a hot take," "actual analysis," "ratio'd by banter"). The distinctions are native to the
community, not imposed from outside.

**Why it fits a classification task.** Because the *same topic* (e.g., "is Player X good?") shows up
as all four discourse types, the model can't cheat by keying on topic words — it has to learn the
**structure and intent** of a comment. That is what makes the analysis-vs-hot-take boundary a real
learning problem rather than a keyword-matching exercise.

---

## 2. Labels (taxonomy)

Four mutually exclusive labels. Together they cover well over 90% of real r/nba comments without an
"other" bucket. The core question for each comment is: **what is this comment *doing*?**

> Note: the starter notebook ships an illustrative 3-label example (analysis / hot_take / reaction).
> This taxonomy is my own design: I sharpened each definition into a stated decision boundary, wrote
> my own examples, and added a fourth label (**banter**) because jokes and trash talk are a dominant,
> distinct mode of r/nba discourse that the 3-label version would force into the wrong bucket.

### `analysis`
> Advances a basketball claim and backs it with **specific, verifiable evidence** — stats, game-film
> detail, lineup/scheme observation, or a concrete historical comparison. If you stripped out the
> opinion, the evidence would still stand on its own.

- *Example 1:* "When Draymond sits, GSW's defensive rating jumps from 108 to 117 and they get killed on the glass — that's why that lineup actually works, it's not just the shooting."
- *Example 2:* "Jokić isn't only scoring. He's at 9.8 assists and the Nuggets are +12 per 100 with him on, −3 off. That on/off swing is the whole offense."

### `hot_take`
> A bold, confident opinion or prediction asserted **without real supporting evidence**. It may drop a
> stat *decoratively*, but the claim rests on conviction, not reasoning. Persuasion by assertion.

- *Example 1:* "Tatum is a top-3 player in the league and it's not even close. Anyone who disagrees doesn't watch basketball."
- *Example 2:* "Lakers are winning the title this year. Book it."

### `reaction`
> An immediate **emotional** response to a specific moment — a play, a trade, a final score, a stat line.
> It expresses a feeling in the moment and makes **no argument**.

- *Example 1:* "WHAT A SHOT. I'm shaking. That's the most clutch thing I've ever seen."
- *Example 2:* "Bro I genuinely can't believe they traded him. I'm heartbroken right now, this sucks."

### `banter`
> Humor, memes, trash talk, copypasta, or one-liners where the point is **comedic or social** rather than
> to advance a sincere basketball claim.

- *Example 1:* "Knicks fans woke up and chose delusion today 💀"
- *Example 2:* "Refs already got the Lakers primetime script printed and laminated I see."

### Mutual-exclusivity check
Every label answers "what is the comment *doing*?" with a different verb: **arguing with evidence**
(analysis), **asserting an opinion** (hot_take), **expressing a feeling** (reaction), **joking**
(banter). A comment can *touch* more than one, but one function almost always dominates — and the
decision rules below resolve the cases where it doesn't.

---

## 3. Hard edge cases (and the decision rules I'll use)

These are the boundaries where two labels genuinely compete. Each has an explicit rule I committed to
*before* annotating, so my labels stay consistent. The first one is the heart of the project.

**A. `hot_take` vs `analysis` — the core boundary.**
*Rule:* If the comment provides specific, verifiable evidence that would still support the claim with the
opinion framing removed → **analysis**. If the evidence is vague, cherry-picked, or decorative (one stat
selected for effect, not as part of reasoning) → **hot_take**.
*Worked case:* "LeBron is overrated — his playoff record vs. top seeds is below .500." One stat, selected
for effect, accusatory framing, no actual reasoning chain → **hot_take**.

**B. `reaction` vs `hot_take`.**
Both can be emotional and unsupported.
*Rule:* If the comment is *expressing a feeling about a specific moment* with no general claim → **reaction**.
If it asserts a *general evaluative claim* ("he's top 3, not close"), even if emotionally charged → **hot_take**.
*Worked case:* "I'M SHAKING THAT WAS INSANE" → reaction. "That shot proves he's the best closer alive" → hot_take.

**C. `banter` vs `hot_take` — the genuinely hardest one (sarcasm).**
A sarcastic, exaggerated comment can read as both a joke and an opinion.
*Rule:* Ask what a regular would say the comment is *for*. If the comedic/social/trash-talk function
dominates → **banter**. If there's a sincere bold opinion underneath the snark → **hot_take**.
*Worked case:* "lol imagine starting this guy in your fantasy league 🤡" → banter (the function is mockery).
"This guy is a fraud and y'all will see it in the playoffs" → hot_take (sincere claim, just rude).

**D. `banter` vs `reaction`.**
*Rule:* A genuine in-the-moment emotional outburst → **reaction**; a performative one-liner played for
laughs → **banter**.

> I'll keep a running log of every comment that made me hesitate (which two labels, what I decided, why)
> in the **Annotation log** section below. The three most instructive ones get written up in the README.

---

## 4. Data collection plan

**Source.** Public r/nba comments only (no login-gated content). I'll read on `old.reddit.com/r/nba`
(easiest to copy plain text from) and pull individual comments — not whole threads — from a mix of
thread types chosen so each label is well represented (full method in `data/HOW_TO_COLLECT.md`):

| Label | Best place to find it on r/nba |
|---|---|
| `analysis` | Top comments on stat posts, `[Highlight]`/film threads, and Post-Game Thread analysis replies |
| `hot_take` | Daily Discussion Thread, player-ranking threads, "unpopular opinion" threads |
| `reaction` | Live Game Threads / Post-Game Threads sorted by Top (in-the-moment) |
| `banter` | Game Threads, rivalry/news threads, anywhere with trash talk |

**How many per label.** Target **~60 per label (≈240 total)** so every class clears the ≥20% floor with
margin and the 15% test split still leaves ~9 examples per class. `analysis` is the scarcest mode on r/nba,
so I'll collect it first and let the others fill in around it.

**If a label is underrepresented after the first pass.** `analysis` is the likely shortfall. Mitigation:
go straight to Post-Game Thread top comments and stat-post comment sections (where substantive comments
concentrate) and keep pulling until `analysis` ≥ 50. I will **not** let any label exceed 70% of the set; if
one drifts high I stop collecting it and backfill the others. Balance over raw count.

**Format.** One CSV — `data/takemeter_nba.csv` — with columns `text`, `label`, `notes`. The notebook does
the 70/15/15 stratified split automatically, so I save a single un-split file. `notes` holds my reason for
any comment that gave me pause.

**Hygiene rules while collecting.** Strip usernames, "Edit:", and quoted parent text; skip comments that are
only a link/image/emoji or that don't make sense without their parent; keep comments roughly 1–4 sentences
(typical comment length); one comment per row.

---

## 5. Evaluation metrics (and why these)

Accuracy alone is not enough here because the classes have different base rates and difficulty — `banter` is
stylistically obvious (a model can score high on it) while the `analysis`/`hot_take` boundary is subtle.
A high overall accuracy could hide a model that has essentially given up on the one distinction the project
is about. So I'll report:

- **Overall accuracy** — headline sanity check, and the number directly comparable to the Groq baseline.
- **Per-class precision / recall / F1** — the real story. I specifically watch **`analysis` recall**
  (does the model actually catch substantive comments, or quietly relabel them `hot_take`?) and
  **`hot_take` precision** (does it over-assign the easy "opinion" bucket?).
- **Macro-F1** — my single headline metric. It averages F1 across classes *equally*, so a class the model
  ignores tanks the score even if that class is small. This is the right summary for a mildly imbalanced,
  uneven-difficulty task.
- **Confusion matrix** — to read the *direction* of errors. I expect the diagnostic cells to be
  `analysis → hot_take` (model misses the evidence) and `banter ↔ reaction` (both short and emotional).

**Baseline.** Zero-shot `llama-3.3-70b-versatile` (Groq) on the same locked test set, scored on the same
metrics, so "did fine-tuning help, and on which classes?" has a concrete answer.

---

## 6. Definition of success

Specific thresholds I set in advance so I can objectively judge the result:

- **Minimum bar (fine-tuning was worth it):** fine-tuned **macro-F1 ≥ 0.60**, it **beats the Groq baseline
  on macro-F1**, and **every class has F1 > 0** (no class is silently ignored).
- **Good enough to deploy as a discourse-quality helper:** **macro-F1 ≥ 0.70**, **`analysis` recall ≥ 0.60**
  (it reliably surfaces substantive comments — the whole point of the tool), and **`banter` is not confused
  with `reaction`/`hot_take`** often enough to be annoying.
- **Stretch / calibration:** predictions made with >0.85 confidence are correct ≥85% of the time, so a
  confidence score could gate an "is this a quality take?" UI.

If the fine-tuned model can't clear the minimum bar, the honest conclusion is either the labels are
inconsistent or 240 examples is too few for the analysis/hot_take boundary — and I'll say so in the report
rather than dress it up.

---

## 7. AI Tool Plan

This project has no application code to generate, so AI assistance shows up in three specific places:

**a) Label stress-testing (done during design).** I gave Claude the four label definitions and edge-case rules
above and asked it to generate ~10 comments that deliberately straddle two labels (especially sarcastic
hot-takes and one-decorative-stat opinions). Cases I couldn't classify cleanly drove the decision rules in
§3 — particularly rule **C** (banter vs hot_take), which only got sharp after seeing borderline sarcasm.

**b) Annotation assistance (optional, with full review).** I may paste batches of ~20 unlabeled comments to an
LLM with the §2 definitions and have it pre-label, then **review and correct every single one** myself before
it enters the dataset. Pre-labeling speeds throughput; it does not replace reading each comment. Any
pre-labeled batch is disclosed in the README's AI-usage section, and I track which rows were pre-labeled.

**c) Failure analysis (after Milestone 5).** I'll paste the fine-tuned model's wrong predictions into Claude
and ask it to surface systematic patterns (label pair, comment length, sarcasm, low-information posts). I'll
then **re-read each cited example myself** to confirm or discard the pattern before it goes in the report.
The LLM proposes; I verify.

---

## 8. Annotation log (hard calls from the real dataset)

> Filled in *during* labeling of the scraped pool. Format: comment → labels in tension → decision → rule.
> The first three go into the README's "difficult to label" section.

1. **"Harper not playing 40 minutes a game will go down as one of the worst coaching decisions of all
   time. Especially when he becomes a perennial all nba player. He is already top10 in rapm."**
   → **analysis vs hot_take.** It pairs hyperbole ("worst of all time") with an actual advanced stat
   (top-10 RAPM). Rule A: a specific, verifiable stat is doing real work in the claim → **analysis**.

2. **"Disgusting trade by the Suns. I absolutely hate this."**
   → **reaction vs hot_take.** "I absolutely hate this" is raw emotion (reaction), but "Disgusting trade"
   is an evaluative judgment about the move. Rule B: it asserts a general evaluation, not just a feeling
   → **hot_take**.

3. **"Castle won the game. That three combined with the two clutch free throws. To hit those shots in the
   NBA Finals at MSG is absolutely cold blooded."**
   → **reaction vs analysis.** Emotional register, but it identifies the *specific plays* that decided the
   game (the three + two FTs). Rule: concrete, verifiable game events as evidence → **analysis**.

4. **"Everyone better than Brunson until it's time to be better than Brunson."** → banter vs hot_take.
   Recurring meme phrasing, but it sincerely asserts Brunson is underrated → **hot_take** (rule C judged the
   sincere claim to dominate the joke).

5. **"We need to have the uncomfortable conversation that KAT is putting up a top 10 advanced stat playoff
   run ever."** → analysis vs hot_take. Claims a stat ranking without citing the numbers → kept **analysis**
   (it points to a specific, checkable advanced-stat claim), but a genuinely borderline call.

---

## 9. Spec changes / updates log

- **Data collection method changed from manual copy-paste to authenticated scraping.** Reddit blocks
  unauthenticated programmatic reads (403), so collection was done via Playwright: a real headed Chromium
  loads r/nba (clearing Reddit's WAF and getting session cookies), then makes same-origin `fetch()` calls
  to the public `.json` endpoints. Scripts: `src/collect_reddit.py` (trade/news threads) and
  `src/collect_postgame.py` (Post-Game/Game threads, added to balance `reaction`). ~1,100 candidate
  comments pooled; 200 selected and labeled (50/label).
- **Labeling done by Claude (Opus), not by the llama-3.3-70b baseline model** — deliberately, so the
  zero-shot baseline stays an independent comparison rather than being graded against its own labels.
  Disclosed in the README AI-usage section.
- **Stretch features attempted:** deployed interface (`app/app.py`). Update this log before adding more.
