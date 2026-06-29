"""
TakeMeter — r/nba comment collector.

Reddit blocks plain HTTP and its `.json` endpoints for non-browser clients
(403 "blocked by network security"). This works around that the legitimate way
a logged-out human browser does: launch a real (headed) Chromium via Playwright,
load the r/nba HTML page once so the WAF issues clearance cookies, then make
same-origin fetch() calls to the public `.json` endpoints from inside that page.

Output: data/_pool_raw.csv  — a deduped pool of candidate comments (text only,
plus score/permalink for reference). Labeling is done separately, by hand.

Usage:  python src/collect_reddit.py
Requires:  pip install playwright ;  python -m playwright install chromium
"""

import csv
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "data" / "_pool_raw.csv"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Listings to sample posts from — a spread so all four discourse types appear.
LISTINGS = [
    "/r/nba/hot.json?limit=40",
    "/r/nba/top.json?t=week&limit=50",
    "/r/nba/top.json?t=month&limit=40",
    "/r/nba/top.json?t=day&limit=30",
]
MAX_POSTS = 70          # cap on threads to open
POOL_TARGET = 600       # stop once we have this many candidate comments
WS = re.compile(r"\s+")


def clean(body: str) -> str:
    body = WS.sub(" ", body).strip()
    return body


def keep(body: str) -> bool:
    if not body or body in ("[deleted]", "[removed]"):
        return False
    if len(body) < 40 or len(body) > 400:
        return False
    if body.startswith(">"):              # quoted parent
        return False
    if body.count("http") and len(body) < 120:   # basically a link
        return False
    low = body.lower()
    if "i am a bot" in low or "your submission was removed" in low:
        return False
    return True


def walk_comments(children, out, depth=0):
    """Collect comment bodies from a Reddit comment listing tree (top 3 levels)."""
    for c in children:
        if c.get("kind") != "t1":
            continue
        d = c.get("data", {})
        if d.get("author") == "AutoModerator" or d.get("stickied"):
            continue
        body = clean(d.get("body", ""))
        if keep(body):
            out.append({"text": body, "score": d.get("score", 0),
                        "permalink": d.get("permalink", "")})
        if depth < 2:
            replies = d.get("replies")
            if isinstance(replies, dict):
                walk_comments(replies.get("data", {}).get("children", []), out, depth + 1)


def page_fetch_json(page, path):
    """Same-origin fetch of a reddit .json path from inside the cleared page."""
    res = page.evaluate(
        """async (p) => {
            const r = await fetch(p, {headers: {'Accept': 'application/json'}});
            return {status: r.status, text: await r.text()};
        }""", path)
    if res["status"] != 200:
        raise RuntimeError(f"status {res['status']} for {path}")
    return json.loads(res["text"])


def main():
    pool, seen = [], set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False,
                                    args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(user_agent=UA, locale="en-US",
                                  viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        print("Loading r/nba to obtain WAF clearance...")
        page.goto("https://www.reddit.com/r/nba/", wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(6000)

        # 1) gather post permalinks from the listings
        permalinks = []
        for lst in LISTINGS:
            try:
                data = page_fetch_json(page, lst)
                for c in data["data"]["children"]:
                    d = c["data"]
                    if d.get("stickied"):
                        continue
                    pl = d.get("permalink")
                    if pl and pl not in seen:
                        seen.add(pl)
                        permalinks.append(pl)
                print(f"  {lst} -> running total posts: {len(permalinks)}")
            except Exception as e:
                print(f"  listing failed {lst}: {e}")
            time.sleep(1.0)

        permalinks = permalinks[:MAX_POSTS]
        print(f"Opening {len(permalinks)} threads for comments...")

        # 2) pull comments per post
        for i, pl in enumerate(permalinks):
            if len(pool) >= POOL_TARGET:
                break
            try:
                data = page_fetch_json(page, f"{pl}.json?limit=100&sort=top")
                if len(data) > 1:
                    walk_comments(data[1]["data"]["children"], pool)
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(permalinks)} threads | pool={len(pool)}")
            except Exception as e:
                print(f"  thread failed {pl}: {e}")
            time.sleep(1.0)

        browser.close()

    # 3) dedupe by text and write
    uniq, texts = [], set()
    for row in pool:
        t = row["text"]
        if t not in texts:
            texts.add(t)
            uniq.append(row)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "score", "permalink"])
        w.writeheader()
        w.writerows(uniq)
    print(f"\nDone. {len(uniq)} unique candidate comments -> {OUT}")


if __name__ == "__main__":
    main()
