"""
reads the raw Yelp Open Dataset JSON files and outputs two smaller CSVs
containing only cleaning-related businesses and their relevant reviews.
https://business.yelp.com/data/resources/open-dataset/
download from above then unzip into some folder and point to it
"""

import json
import csv
import os
from datetime import datetime, timedelta

# this is my local path so change to yours
BUSINESS_FILE = r"C:\CSULB\SmartClean-AI\yelp dataset\yelp_academic_dataset_business.json"
REVIEW_FILE   = r"C:\CSULB\SmartClean-AI\yelp dataset\yelp_academic_dataset_review.json"

OUT_DIR           = os.path.dirname(os.path.abspath(__file__))
OUT_BUSINESSES    = os.path.join(OUT_DIR, "yelp_businesses_clean.csv")
OUT_REVIEWS       = os.path.join(OUT_DIR, "yelp_reviews_clean.csv")

CLEANING_KEYWORDS = [
    "home cleaning",
    "house cleaning",
    "maid service",
    "office cleaning",
    "carpet cleaning",
    "window washing",
    "window cleaning",
    "housekeeping",
    "janitorial",
    "pressure wash",
    "move-in/move-out",
    "deep cleaning",
    "cleaning service",
]

# Reviews older than this are excluded (5 years back from a ~2022 dataset snapshot)
REVIEW_CUTOFF = datetime(2017, 1, 1)

# Reviews shorter than this (characters) are excluded as uninformative one-liners
MIN_REVIEW_LENGTH = 150

# Progress reporting interval (lines read)
PROGRESS_EVERY = 100_000

def is_cleaning_business(categories: str | None) -> bool:
    if not categories:
        return False
    cats_lower = categories.lower()
    return any(kw in cats_lower for kw in CLEANING_KEYWORDS)


def parse_date(date_str: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str[:19], fmt)
        except ValueError:
            continue
    return None

def filter_businesses() -> set[str]:
    print(f"\n[Step 1] Filtering businesses from:\n  {BUSINESS_FILE}")

    business_cols = [
        "business_id", "name", "city", "state",
        "stars", "review_count", "categories", "is_open",
    ]

    kept_ids: set[str] = set()
    kept = 0
    total = 0

    with open(BUSINESS_FILE, "r", encoding="utf-8") as fin, \
         open(OUT_BUSINESSES, "w", newline="", encoding="utf-8") as fout:

        writer = csv.DictWriter(fout, fieldnames=business_cols, extrasaction="ignore")
        writer.writeheader()

        for line in fin:
            total += 1
            if total % PROGRESS_EVERY == 0:
                print(f"  ... {total:,} businesses read, {kept:,} matched")

            line = line.strip()
            if not line:
                continue

            try:
                biz = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not is_cleaning_business(biz.get("categories")):
                continue

            writer.writerow(biz)
            kept_ids.add(biz["business_id"])
            kept += 1

    print(f"  Done. {kept:,} cleaning businesses saved -> {OUT_BUSINESSES}")
    return kept_ids

def filter_reviews(business_ids: set[str]) -> None:
    print(f"\n[Step 2] Filtering reviews from:\n  {REVIEW_FILE}")
    print(f"  Criteria: business in filtered set · date >= {REVIEW_CUTOFF.date()} "
          f"· length >= {MIN_REVIEW_LENGTH} chars")

    review_cols = ["review_id", "business_id", "stars", "date", "text"]

    kept = 0
    skipped_biz = 0
    skipped_date = 0
    skipped_len = 0
    total = 0

    with open(REVIEW_FILE, "r", encoding="utf-8") as fin, \
         open(OUT_REVIEWS, "w", newline="", encoding="utf-8") as fout:

        writer = csv.DictWriter(fout, fieldnames=review_cols, extrasaction="ignore")
        writer.writeheader()

        for line in fin:
            total += 1
            if total % PROGRESS_EVERY == 0:
                print(f"  ... {total:,} reviews read, {kept:,} kept")

            line = line.strip()
            if not line:
                continue

            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Business filter
            if review.get("business_id") not in business_ids:
                skipped_biz += 1
                continue

            # Date filter
            '''
            review_date = parse_date(review.get("date", ""))
            if review_date is None or review_date < REVIEW_CUTOFF:
                skipped_date += 1
                continue
            '''

            # Length filter
            text = review.get("text", "")
            if len(text) < MIN_REVIEW_LENGTH:
                skipped_len += 1
                continue

            writer.writerow(review)
            kept += 1

    print(f"\n  Done.")
    print(f"  {kept:,} reviews saved -> {OUT_REVIEWS}")
    print(f"  Skipped - wrong business: {skipped_biz:,}, "
          f"too old: {skipped_date:,}, too short: {skipped_len:,}")

if __name__ == "__main__":
    for path, label in [(BUSINESS_FILE, "business"), (REVIEW_FILE, "review")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Yelp {label} file not found:\n  {path}\n"
                "Update the path at the top of this script."
            )

    business_ids = filter_businesses()
    filter_reviews(business_ids)

    print("\nAll done. Next step: run data/tag_cleaners.py to extract LLM tags.")
