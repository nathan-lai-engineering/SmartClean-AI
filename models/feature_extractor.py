import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

# match 12 tags in read me
TAG_COLUMNS = [
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


# extract number info
def extract_number(keyword, text):
    match = re.search(rf'(\d+)\s*(?:\w+\s*){{0,3}}{keyword}', text)
    return int(match.group(1)) if match else None


def estimate_hours(gr_liv_area, bedroom_abvgr, full_bath, house_age):
    return (
        1.5
        + 0.0015 * gr_liv_area
        + 0.4   * bedroom_abvgr
        + 0.6   * full_bath
        + 0.003 * house_age
    )


def derive_job_type(features: dict) -> str:
    if features["move_out"]:          return "move_out"
    if features["deep_clean"]:        return "deep_clean"
    if features["post_construction"]: return "post_construction"
    if features["fast_turnaround"]:   return "fast_turnaround"
    return "standard"


# extract features from user inputs
def extract_features(text: str) -> dict:
    t = text.lower()

    features = {tag: 0 for tag in TAG_COLUMNS}

    if "deep clean" in t:
        features["deep_clean"] = 1
    if "move out" in t:
        features["move_out"] = 1
    if "construction" in t:
        features["post_construction"] = 1
    if "eco" in t:
        features["eco_friendly"] = 1
    if "pet" in t:
        features["pet_friendly"] = 1
    if "window" in t:
        features["window_cleaning"] = 1
    if "office" in t or "commercial" in t:
        features["office_commercial"] = 1
    if "detail oriented" in t or "detailed oriented" in t:
        features["detail_oriented"] = 1
    if "urgent" in t or "fast" in t:
        features["fast_turnaround"] = 1
    if "reliable" in t:
        features["reliable"] = 1
    if "communicative" in t:
        features["communicative"] = 1
    if "experienced" in t:
        features["experienced"] = 1

    bedrooms  = extract_number("bed", t)
    full_bath = extract_number("full bath", t) or extract_number("bath", t)

    sq_ft = (extract_number("sq ft", t)
             or extract_number("sqft", t)
             or extract_number("square", t))

    house_age = extract_number("year", t)
    if house_age and house_age > 1900:
        house_age = 2026 - house_age

    budget_match = re.search(r'\$\s*(\d+)', t) or re.search(r'(\d+)\s*(?:per hour|/hr|an hour)', t)
    budget = float(budget_match.group(1)) if budget_match else None

    features["bedroom_abvgr"]          = bedrooms  if bedrooms  is not None else 3
    features["full_bath"]              = full_bath if full_bath is not None else 2
    features["gr_liv_area"]            = sq_ft     if sq_ft     is not None else 1500
    features["house_age"]              = house_age if house_age is not None else 20
    features["target_budget_per_hour"] = budget    if budget    is not None else 45.0

    features["job_type"]        = derive_job_type(features)
    features["estimated_hours"] = estimate_hours(
        features["gr_liv_area"], features["bedroom_abvgr"],
        features["full_bath"],   features["house_age"]
    )

    return features


GROQ_PROMPT = """Extract cleaning job details from the customer request below.
Return ONLY a valid JSON object with exactly these keys and no others:

{{
  "job_type": one of "standard", "deep_clean", "move_out", "post_construction",
  "gr_liv_area": integer square footage (default 1500),
  "bedroom_abvgr": integer bedrooms (default 3),
  "full_bath": integer full bathrooms (default 2),
  "house_age": integer approximate home age in years (default 20),
  "target_budget_per_hour": float $/hr budget (default 45.0),
  "deep_clean": 0 or 1,
  "move_out": 0 or 1,
  "post_construction": 0 or 1,
  "eco_friendly": 0 or 1,
  "pet_friendly": 0 or 1,
  "window_cleaning": 0 or 1,
  "office_commercial": 0 or 1,
  "detail_oriented": 0 or 1,
  "fast_turnaround": 0 or 1,
  "reliable": 0 or 1,
  "communicative": 0 or 1,
  "experienced": 0 or 1
}}

Rules:
- Infer values from context. Use defaults for anything not mentioned.
- Set deep_clean=1 if job_type is deep_clean. Set move_out=1 if job_type is move_out. Set post_construction=1 if job_type is post_construction.
- reliable, communicative, experienced: set to 1 only if the customer explicitly mentions wanting those qualities.
- Return only the JSON, no explanation.

Customer request:
{request}
"""


def extract_features_groq(text: str) -> dict:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": GROQ_PROMPT.format(request=text)}],
        temperature=0,
        max_tokens=400,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    features = json.loads(raw)
    features["estimated_hours"] = estimate_hours(
        features.get("gr_liv_area", 1500), features.get("bedroom_abvgr", 3),
        features.get("full_bath", 2),      features.get("house_age", 20)
    )
    return features


# manual input from user
def main():
    print("Enter your cleaning request: ")
    user_input = input()

    features = extract_features(user_input)

    print("Extracted Features:")
    print(features)


if __name__ == "__main__":
    main()
