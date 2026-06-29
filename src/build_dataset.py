"""
Assemble the final labeled dataset from the indexed pool.

Labels were assigned by hand (by me, Claude/Opus — deliberately NOT by the
llama-3.3-70b model used as the zero-shot baseline, so the baseline stays an
independent comparison). Each list below holds the pool indices I assigned to
that label after reading the comment. The builder validates the four sets are
disjoint + in range, looks up the exact comment text, and writes the CSV the
notebook expects: text, label, notes.
"""

import csv
import html
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
POOL = DATA / "_pool_all.csv"
OUT = DATA / "takemeter_nba.csv"


def clean(t: str) -> str:
    """Unescape HTML entities and strip Reddit markdown escapes/artifacts."""
    t = html.unescape(t)                 # &amp; &gt; &#39; -> & > '
    t = re.sub(r"\\([-*>&#_~`.])", r"\1", t)   # \-  \*  \>  -> - * >
    t = t.replace("~~", "")              # strikethrough markers
    t = re.sub(r"\s+", " ", t).strip()
    return t

ANALYSIS = [
    10, 39, 41, 98, 123, 125, 133, 134, 154, 157, 189, 198, 238, 259, 280, 285,
    308, 312, 318, 320, 326, 327, 328, 329, 331, 334, 343, 625, 645, 656, 711,
    741, 754, 799, 800, 823, 839, 856, 862, 868, 875, 880, 903, 916, 924, 942,
    959, 974, 976, 977,
]
HOT_TAKE = [
    3, 8, 17, 27, 55, 56, 58, 89, 143, 167, 174, 177, 180, 184, 206, 274, 311,
    332, 333, 341, 349, 357, 360, 635, 689, 690, 692, 693, 697, 703, 712, 714,
    729, 734, 735, 758, 807, 887, 904, 906, 907, 908, 923, 935, 957, 988, 989,
    990, 991, 996,
]
REACTION = [
    50, 136, 638, 639, 651, 660, 666, 677, 694, 696, 717, 718, 719, 727, 736,
    746, 748, 752, 775, 789, 797, 798, 808, 814, 815, 816, 818, 822, 832, 833,
    853, 854, 876, 877, 889, 890, 891, 894, 909, 910, 911, 920, 930, 938, 958,
    962, 971, 985, 986, 993,
]
BANTER = [
    0, 7, 16, 18, 19, 28, 31, 32, 34, 36, 48, 61, 64, 68, 69, 87, 93, 94, 101,
    102, 105, 107, 110, 113, 115, 632, 633, 671, 721, 723, 743, 744, 747, 779,
    805, 811, 812, 836, 865, 866, 881, 918, 940, 952, 965, 967, 979, 981, 999,
    1000,
]

# Notes for genuinely borderline calls (the rest get an empty note).
NOTES = {
    143: "borderline reaction (emotional) vs hot_take; judged a claim on the trade -> hot_take (rule B)",
    711: "has a stat (rapm) but framed as hyperbole ('worst decision ever'); kept analysis per rule A",
    903: "advanced-stat claim w/o the numbers shown; borderline hot_take, kept analysis (cites a stat type)",
    868: "borderline reaction; cites the specific plays (3 + 2 FTs) so -> analysis",
    924: "reasoning not stats; borderline hot_take, kept analysis (concrete clock-management argument)",
    758: "recurring meme phrasing; borderline banter, kept hot_take (sincere claim about Brunson)",
}

LABELS = [("analysis", ANALYSIS), ("hot_take", HOT_TAKE),
          ("reaction", REACTION), ("banter", BANTER)]


def main():
    pool = {}
    with open(POOL, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pool[int(r["idx"])] = r["text"]

    # validate disjoint + in range
    seen = {}
    for label, idxs in LABELS:
        for i in idxs:
            assert i in pool, f"index {i} not in pool"
            assert i not in seen, f"index {i} double-labeled ({seen[i]} and {label})"
            seen[i] = label

    rows = []
    for label, idxs in LABELS:
        for i in idxs:
            rows.append({"text": clean(pool[i]), "label": label, "notes": NOTES.get(i, "")})

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "notes"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} labeled rows -> {OUT}")
    for label, idxs in LABELS:
        print(f"  {label:9} {len(idxs)}")


if __name__ == "__main__":
    main()
