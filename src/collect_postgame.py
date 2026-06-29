"""
Supplementary collection: Post-Game Threads + Game Threads.

The main pool (collect_reddit.py) over-samples trade-news threads, which are
heavy on banter/hot_take but thin on in-the-moment `reaction`. Post-Game and
Game threads are reaction-rich, so this targets them via subreddit search to
balance the label distribution. Same WAF-clearance trick as the main collector.

Output: data/_pool_postgame.csv
"""

import csv
import json
import re
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "data" / "_pool_postgame.csv"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
SEARCHES = [
    "/r/nba/search.json?q=flair%3A%22Post%20Game%20Thread%22&restrict_sr=1&sort=new&t=month&limit=40",
    "/r/nba/search.json?q=flair%3A%22Game%20Thread%22&restrict_sr=1&sort=new&t=month&limit=25",
]
MAX_POSTS = 40
POOL_TARGET = 450
WS = re.compile(r"\s+")


def clean(b): return WS.sub(" ", b).strip()


def keep(b):
    if not b or b in ("[deleted]", "[removed]"):
        return False
    if len(b) < 30 or len(b) > 400:
        return False
    if b.startswith(">"):
        return False
    if b.count("http") and len(b) < 120:
        return False
    low = b.lower()
    return not ("i am a bot" in low or "submission was removed" in low)


def walk(children, out, depth=0):
    for c in children:
        if c.get("kind") != "t1":
            continue
        d = c.get("data", {})
        if d.get("author") == "AutoModerator" or d.get("stickied"):
            continue
        b = clean(d.get("body", ""))
        if keep(b):
            out.append({"text": b, "score": d.get("score", 0), "permalink": d.get("permalink", "")})
        if depth < 1:
            r = d.get("replies")
            if isinstance(r, dict):
                walk(r.get("data", {}).get("children", []), out, depth + 1)


def fetch_json(page, path):
    res = page.evaluate(
        """async (p)=>{const r=await fetch(p,{headers:{'Accept':'application/json'}});return{status:r.status,text:await r.text()};}""",
        path)
    if res["status"] != 200:
        raise RuntimeError(f"status {res['status']}")
    return json.loads(res["text"])


def main():
    pool = []
    with sync_playwright() as p:
        b = p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
        ctx = b.new_context(user_agent=UA, locale="en-US", viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        page.goto("https://www.reddit.com/r/nba/", wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(6000)

        perms, seen = [], set()
        for s in SEARCHES:
            try:
                data = fetch_json(page, s)
                for c in data["data"]["children"]:
                    pl = c["data"].get("permalink")
                    if pl and pl not in seen:
                        seen.add(pl)
                        perms.append(pl)
                print(f"  search -> total posts: {len(perms)}")
            except Exception as e:
                print("  search failed:", e)
            time.sleep(1.0)

        perms = perms[:MAX_POSTS]
        print(f"Opening {len(perms)} post/game threads...")
        for i, pl in enumerate(perms):
            if len(pool) >= POOL_TARGET:
                break
            try:
                data = fetch_json(page, f"{pl}.json?limit=100&sort=top")
                if len(data) > 1:
                    walk(data[1]["data"]["children"], pool)
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(perms)} | pool={len(pool)}")
            except Exception as e:
                print("  thread failed:", e)
            time.sleep(1.0)
        b.close()

    uniq, texts = [], set()
    for r in pool:
        if r["text"] not in texts:
            texts.add(r["text"])
            uniq.append(r)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "score", "permalink"])
        w.writeheader()
        w.writerows(uniq)
    print(f"\nDone. {len(uniq)} unique comments -> {OUT}")


if __name__ == "__main__":
    main()
