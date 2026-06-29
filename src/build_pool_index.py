"""Merge + dedupe the two raw pools into one indexed file for hand-labeling."""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SRCS = [DATA / "_pool_raw.csv", DATA / "_pool_postgame.csv"]
OUT = DATA / "_pool_all.csv"

rows, seen = [], set()
for src in SRCS:
    if not src.exists():
        continue
    with open(src, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = r["text"].strip()
            key = t.lower()
            if not t or key in seen:
                continue
            seen.add(key)
            rows.append({"idx": len(rows), "text": t,
                         "score": r.get("score", ""), "permalink": r.get("permalink", "")})

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["idx", "text", "score", "permalink"])
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} unique comments -> {OUT}")
