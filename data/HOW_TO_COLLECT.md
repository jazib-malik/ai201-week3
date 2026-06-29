# How to Collect Your 240 r/nba Comments

This is your one manual job for the dataset. Budget **1–2 hours**. The goal is **~240 real public
comments** (about **60 per label**), saved into `takemeter_nba.csv`. Don't build a scraper — manual
collection is faster here and keeps you close to the data (which is the point).

---

## The fastest workflow (recommended)

1. **Open a Google Sheet** (or Excel) with three columns: `text` | `label` | `notes`.
   Collect into the sheet, then **File → Download → CSV** at the end. (Spreadsheets handle commas and
   quotes inside comments for you — much less error-prone than typing CSV by hand.)
2. **Browse r/nba on `old.reddit.com/r/nba`** — it's plain HTML and easy to select/copy comment text.
   No login needed to read.
3. For each comment you grab:
   - Paste the comment text into `text`.
   - Decide its label using the definitions in `planning.md` §2 and the decision rules in §3.
   - If you hesitated, jot which two labels were in tension and what you decided in `notes`.
4. Keep a tally per label. Stop a label at ~60; keep filling the others.

---

## Where to find each label (so you stay balanced)

`analysis` is the scarcest — **collect it first** so you don't run out of room.

| Label | Where to look | Sort by |
|---|---|---|
| **analysis** | Top comments on **stat posts**, `[Highlight]`/film-breakdown threads, and the analysis replies in **Post-Game Threads**. Look for on/off numbers, scheme talk, historical comps. | Top |
| **hot_take** | **Daily Discussion Thread**, player-ranking threads, "unpopular NBA opinion" threads, tier-list posts. | Top / Controversial |
| **reaction** | **Live Game Threads** and **Post-Game Threads** during/after big games — in-the-moment "I'M SHAKING" energy. | Top / New (during a game) |
| **banter** | Game Threads, rivalry threads, news threads. Trash talk, memes, jokes, one-liners. | Top |

Tip: open 2–3 recent Post-Game Threads and one Daily Discussion Thread and you'll hit all four labels
quickly.

---

## What makes a good row (collection hygiene)

**Keep:**
- Self-contained comments, roughly **1–4 sentences** (typical comment length).
- Comments that make sense on their own.

**Clean up before pasting:**
- Strip the username, vote counts, "Edit:", and any quoted parent text.
- Replace internal line breaks with a space so each comment is one row.

**Skip:**
- Comments that are only a link, image, GIF, or emoji.
- Comments that only make sense as a reply to a parent ("^ this", "lol exactly").
- Anything with personal info or that names/attacks a specific non-public user.

---

## Balance check (do this before you stop)

Run a quick `value_counts()` (the notebook prints this in Section 1) or just eyeball your sheet:

- Aim for **~60 per label**; **no label below ~50 (≈20%)** and **none above 70%** of the set.
- If `analysis` is short, go back to stat-post and Post-Game top comments and pull only `analysis`
  until it clears 50. Backfill rather than letting `banter`/`reaction` run away.

---

## Optional speed-up (disclose it if you use it)

You can paste a batch of ~20 unlabeled comments into Claude or Groq with the §2 label definitions and
ask it to pre-label them, **then review and correct every one yourself** before they go in the CSV.
This is allowed but must be disclosed in the README's AI-usage section, and skimming defeats the
purpose — read each comment. Mark pre-labeled rows in `notes` (e.g. `prelabeled`) so you can track them.

---

## When you're done

Save/replace `takemeter_nba.csv` in this folder with your real rows (delete the EXAMPLE rows that ship
in the template). Then follow the steps in the repo's `README.md` → "How to run" section.
