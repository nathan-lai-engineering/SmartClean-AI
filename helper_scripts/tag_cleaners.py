"""
Reads yelp_businesses_clean.csv + yelp_reviews_clean.csv.
Calls Groq (Llama 3.1) to extract 12 fixed capability tags per business.
"""

import os
import re
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(ROOT_DIR, "data")
BUSINESSES_CSV = os.path.join(DATA_DIR, "yelp_businesses_clean.csv")
REVIEWS_CSV    = os.path.join(DATA_DIR, "yelp_reviews_clean.csv")
OUTPUT_CSV     = os.path.join(DATA_DIR, "demo_cleaner_tags.csv")

SAMPLE_SIZE    = 200
MAX_REVIEWS    = 20   # reviews per business sent to LLM
TOKENS_PER_MINUTE = 6000  # Groq free tier limit for llama-3.3-70b-versatile
RATE_LIMIT_BUFFER = 1.15  # 15% safety margin

TAGS = [
    "deep_clean",
    "move_out",
    "post_construction",
    "eco_friendly",
    "pet_friendly",
    "window_cleaning",
    "office_commercial",
    "detail_oriented",
    "fast_turnaround",
    "reliable",
    "communicative",
    "experienced",
]

PROMPT_TEMPLATE = """You are tagging a cleaning business based on customer reviews.

Return ONLY a comma-separated list of tags that clearly apply, chosen from this fixed list:
deep_clean, move_out, post_construction, eco_friendly, pet_friendly, window_cleaning, office_commercial, detail_oriented, fast_turnaround, reliable, communicative, experienced

Rules:
- Only include tags with clear evidence in the reviews.
- Do not include any tags not on the list.
- Do not explain or add any other text.

Reviews:
{reviews}
"""

def extract_tags(client, reviews_text):
    prompt = PROMPT_TEMPLATE.format(reviews=reviews_text)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip().lower()
    found = set(re.findall(r"[a-z_]+", raw))
    tags = {tag: int(tag in found) for tag in TAGS}
    tokens_used = response.usage.total_tokens
    return tags, tokens_used


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set.")
    client = Groq(api_key=api_key)

    businesses = pd.read_csv(BUSINESSES_CSV)
    reviews    = pd.read_csv(REVIEWS_CSV)

    # keep only businesses that have at least one review
    has_reviews = set(reviews["business_id"].unique())
    pool = businesses[businesses["business_id"].isin(has_reviews)].copy()

    sample = pool.sample(n=min(SAMPLE_SIZE, len(pool)), random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sample)} businesses.")

    # load already-tagged businesses to skip them
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        already_tagged = set(existing["business_id"].tolist())
        results = existing.to_dict("records")
        print(f"Resuming: {len(already_tagged)} already tagged, {len(sample) - len(already_tagged)} remaining.")
    else:
        already_tagged = set()
        results = []

    for i, row in sample.iterrows():
        bid  = row["business_id"]
        name = row["name"]

        biz_reviews = reviews[reviews["business_id"] == bid]["text"].tolist()
        biz_reviews = biz_reviews[:MAX_REVIEWS]
        reviews_text = "\n\n".join(biz_reviews)

        if bid in already_tagged:
            continue

        print(f"[{i+1}/{len(sample)}] {name} ({len(biz_reviews)} reviews)...")

        t0 = time.time()
        try:
            tags, tokens_used = extract_tags(client, reviews_text)
        except Exception as e:
            print(f"  ERROR: {e} -- skipping, all tags set to 0")
            tags = {tag: 0 for tag in TAGS}
            tokens_used = 0

        entry = {"business_id": bid, "name": name}
        entry.update(tags)
        results.append(entry)

        # adaptive delay: wait long enough so this request's tokens don't exceed TPM
        elapsed = time.time() - t0
        required = (tokens_used / TOKENS_PER_MINUTE) * 60 * RATE_LIMIT_BUFFER
        wait = max(0, required - elapsed)
        print(f"  tokens: {tokens_used}, wait: {wait:.1f}s")
        time.sleep(wait)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
    print(df[TAGS].sum().to_string())


if __name__ == "__main__":
    main()
