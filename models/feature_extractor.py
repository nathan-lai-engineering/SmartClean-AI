import re

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
    match = re.search(rf'(\d+).*{keyword}', text)
    return int(match.group(1)) if match else None

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
    if "detailed oriented" in t:
        features["detail_oriented"] = 1
    if "urgent" in t or "fast" in t:
        features["fast_turnaround"] = 1
    if "reliable" in t:
        features["reliable"] = 1
    if "communicative" in t:
        features["communicative"] = 1
    if "experienced" in t:
        features["experienced"] = 1

    bedrooms = extract_number("bed", t)
    bathrooms = extract_number("bath", t)

    budget_match = re.search(r'\d+', t)
    budget = float(budget_match.group()) if budget_match else None

    features["bedroom_abvgr"] = bedrooms if bedrooms is not None else 3
    features["full_bath"] = bathrooms if bathrooms is not None else 2
    features["target_budget_per_hour"] = budget if budget is not None else 45.0

    return features

# test
def main():
    test_text = "Need a deep clean for a 2 bed 2 bath apartment with a dog. Budget is $50/hr"
    features = extract_features(test_text)
    print(features)


if __name__ == "__main__":
    main()