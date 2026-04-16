# SmartClean AI

**Team 4:** Dante Berouty, Nathan Lai, Yongyan Liang
**Course:** CECS 551 - Phase 3 Prototype

---

## Background

Traditional cleaning platforms like MaidPro, Handy, and Rise & Shine Cleaning assign cleaners based on availability alone, with no consideration for the customer's specific needs, the cleaner's strengths, or past performance. This creates problems on both sides:

- Customers get inconsistent quality such as a landlord needing fast turnovers gets the same result as a family needing meticulous deep cleaning
- Cleaners get mismatched jobs such as being sent to jobs outside their skill set costs them time and reputation
- Platforms take on the financial risk of a bad match such as refunds, bad reviews, lost customers

Target customers are landlords, busy homeowners, and short-term rental managers with an estimated 1,000–1,500 households in the immediate area who need reliable, repeated service and have the most to gain from consistent matching quality.

No existing platform learns from past jobs, customer feedback, or cleaner specializations to improve future matches. That's the gap SmartClean AI addresses.

---

## System Design Overview

SmartClean AI uses a machine learning recommendation engine to match customers to the best-fit cleaner, and not just the nearest available one. The system pulls features from both sides of the match including what the job needs and what the cleaner is good at, and scores every possible pairing.

**End-to-end pipeline:**

```
Customer Request (free text or form)
        |
        v
NLP Feature Extractor
(regex + keyword parsing -> home_size, cleaning_type, budget, special_requirements)
        |
        v
Cleaner Profile Pool (real Yelp businesses + LLM-extracted capability tags)
        |
        v
Random Forest Regressor
(predicts compatibility score 0-100 for each job-cleaner pair)
        |
        v
Ranked Recommendations (Streamlit dashboard)
top matches with score, matched tags, budget status
```

---

## Key Decisions and Business Logic

### Data Sources

**Cleaner side - [Yelp Open Dataset](https://www.yelp.com/dataset)**
- Filtered from 150k+ businesses down to cleaning-related categories (home cleaning, maid service, carpet cleaning, etc.)
- Reviews filtered to past 5 years, minimum 150 characters to remove one-liners
- 50 businesses sampled for the demo pool

**Cleaner rates - [BLS OES 2024, SOC 37-2012](https://www.bls.gov/oes/tables.htm)**
- State-level mean hourly wages for Maids and Housekeeping Cleaners
- Worker wages (~$12-25/hr by state), converted to customer-facing rates with a 2.5x markup + noise (~$30-65/hr)

**Home profiles - [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)**
- Used as a home profile generator: sq footage, bedroom/bathroom counts, year built, quality score
- Augmented with ISSA cleaning time guidelines to estimate cleaning duration per job

**Match labels - Synthetic**
- No public dataset exists for actual job-cleaner match outcomes (proprietary to platforms)
- Labels generated from a weighted formula: specialisation match (30 pts), cleaner rating (20 pts), budget fit (10 pts), experience (10 pts), review volume (5 pts) + Gaussian noise

### LLM Component - [Groq API](https://console.groq.com) (free tier)

Llama 3.1 via Groq reads each Yelp business's filtered reviews and extracts a fixed set of 12 capability tags. Fixed vocabulary keeps profiles consistent and comparable across all cleaners.

- **7 specialisation tags:** deep_clean, move_out, post_construction, eco_friendly, pet_friendly, window_cleaning, office_commercial
- **5 work style tags:** detail_oriented, fast_turnaround, reliable, communicative, experienced

### Matching Model - Random Forest Regressor

- Chosen over simple cosine similarity because it learns non-linear feature interactions and produces explainable feature importances
- 23-dimensional feature vector per job-cleaner pair: job encodings, cleaner profile attributes, derived interaction features (specialisation match fraction, budget ratio, binary spec flags)
- Trained on synthetic job-cleaner pairs, replaceable with real post-service ratings as users accumulate

### What is Real vs. Synthetic

| Component | Source |
|---|---|
| Cleaner business names, cities, ratings, review counts | Real (Yelp) |
| Cleaner capability tags | Real (LLM-extracted from Yelp reviews) |
| Home / job features | Real (Ames Housing Dataset) |
| Cleaner hourly rates | Real baseline (BLS) + synthetic noise |
| Match outcome labels | Synthetic (no public dataset exists) |

---

## Repository Structure

```
SmartClean-AI/
|
+-- data/
|   +-- filter_yelp.py             # filters Yelp dataset to cleaning businesses/reviews
|   +-- tag_cleaners.py            # Groq/Llama extracts 12 tags per business from reviews
|   +-- build_profiles.py          # combines Yelp + wages + tags -> cleaner_profiles.csv
|   +-- generate_training.py       # creates synthetic job-cleaner pairs with satisfaction labels
|   +-- yelp_businesses_clean.csv  (filtered, real data)
|   +-- yelp_reviews_clean.csv     (filtered, real data)
|   +-- state_wages.csv            (real BLS data)
|   +-- cleaner_profiles.csv       (generated)
|   +-- training_data.csv          (generated)
|
+-- models/
|   +-- feature_extractor.py       # NLP: customer request text -> structured feature dict
|   +-- matching_model.py          # ML: train, load, rank_cleaners()
|   +-- trained_model.pkl          (generated on first run)
|
+-- app/
|   +-- main.py                    # Streamlit dashboard
|
+-- docs/
|   +-- project_plan.md
|
+-- requirements.txt
```

---

## Pipeline Run Order

```bash
# 1. One-time data prep (run once)
python data/filter_yelp.py          # set Yelp file paths at top of script
python data/tag_cleaners.py         # requires GROQ_API_KEY env variable
python data/build_profiles.py       # combines all data -> cleaner_profiles.csv
python data/generate_training.py    # creates training_data.csv

# 2. Launch app
streamlit run app/main.py           # trains model on first run, then loads from cache
```
