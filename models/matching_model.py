import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Paths for model file and datasets
MODEL_PATH = "models/matching_model.joblib"
TRAINING_DATA_PATH = "data/training_data.csv"
CLEANER_PROFILES_PATH = "data/cleaner_profiles.csv"

# Features used for training (job + cleaner attributes)
FEATURE_COLUMNS = [
    "gr_liv_area",
    "bedroom_abvgr",
    "full_bath",
    "half_bath",
    "house_age",
    "estimated_hours",
    "target_budget_per_hour",
    "needs_deep_clean",
    "needs_move_out",
    "needs_pet_friendly",
    "needs_fast_turnaround",
    "needs_detail_oriented",
    "needs_eco_friendly",
    "cleaner_rating",
    "cleaner_review_count",
    "hourly_rate_est",
    "deep_clean",
    "move_out",
    "pet_friendly",
    "fast_turnaround",
    "detail_oriented",
    "eco_friendly",
    "reliable",
    "communicative",
    "experienced",
]

TARGET_COLUMN = "compatibility_score"


# Train Random Forest model on synthetic job cleaner dataset
def train_model(
    training_data_path: str = TRAINING_DATA_PATH,
    model_path: str = MODEL_PATH,
):
    df = pd.read_csv(training_data_path).copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize Random Forest with tuned parameters
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate model performance (MSE and R2)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "mse": mse,
            "r2": r2,
        },
        model_path,
    )

    return model, {"mse": mse, "r2": r2}


# Load existing model or train a new one if missing
def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        model, metrics = train_model(model_path=model_path)
        return model, FEATURE_COLUMNS, metrics

    artifact = joblib.load(model_path)
    return artifact["model"], artifact["feature_columns"], {
        "mse": artifact.get("mse"),
        "r2": artifact.get("r2"),
    }

# Standardize user job input to match model feature format
def _normalize_job_features(job: dict) -> dict:
    job = dict(job)

    # Defaults for fields that may be missing from UI/extractor
    job.setdefault("half_bath", 0)

    # Map app/extractor flags to training feature names
    job_features = {
        "gr_liv_area": job.get("gr_liv_area", 1500),
        "bedroom_abvgr": job.get("bedroom_abvgr", 3),
        "full_bath": job.get("full_bath", 2),
        "half_bath": job.get("half_bath", 0),
        "house_age": job.get("house_age", 20),
        "estimated_hours": job.get("estimated_hours", 4.0),
        "target_budget_per_hour": job.get("target_budget_per_hour", 45.0),
        "needs_deep_clean": int(job.get("deep_clean", 0)),
        "needs_move_out": int(job.get("move_out", 0)),
        "needs_pet_friendly": int(job.get("pet_friendly", 0)),
        "needs_fast_turnaround": int(job.get("fast_turnaround", 0)),
        "needs_detail_oriented": int(job.get("detail_oriented", 0)),
        "needs_eco_friendly": int(job.get("eco_friendly", 0)),
    }

    return job_features


# Create job cleaner combinations for scoring
def _build_candidate_frame(job: dict, cleaner_profiles_path: str = CLEANER_PROFILES_PATH):
    cleaners = pd.read_csv(cleaner_profiles_path).copy()
    job_features = _normalize_job_features(job)

    candidates = cleaners.copy()

    candidates["cleaner_rating"] = candidates["stars"]
    candidates["cleaner_review_count"] = candidates["review_count"]

    for col, value in job_features.items():
        candidates[col] = value

    return candidates


# Generate simple explanations for why a cleaner matches the job
def _reason_tags(row: pd.Series, job: dict) -> list[str]:
    reasons = []

    if job.get("deep_clean", 0) and row.get("deep_clean", 0) == 1:
        reasons.append("matches deep-clean needs")
    if job.get("move_out", 0) and row.get("move_out", 0) == 1:
        reasons.append("fits move-out jobs")
    if job.get("pet_friendly", 0) and row.get("pet_friendly", 0) == 1:
        reasons.append("pet-friendly")
    if job.get("fast_turnaround", 0) and row.get("fast_turnaround", 0) == 1:
        reasons.append("fast turnaround")
    if job.get("detail_oriented", 0) and row.get("detail_oriented", 0) == 1:
        reasons.append("detail-oriented")
    if job.get("eco_friendly", 0) and row.get("eco_friendly", 0) == 1:
        reasons.append("eco-friendly")

    if row.get("stars", 0) >= 4.5:
        reasons.append("high rating")
    if abs(row.get("hourly_rate_est", 0) - job.get("target_budget_per_hour", 45.0)) <= 8:
        reasons.append("budget-aligned")

    return reasons[:3]


# Predict compatibility scores and return top cleaner matches
def rank_cleaners(job: dict, top_n: int = 5):
    # Load trained model and feature schema
    model, feature_columns, metrics = load_model()
    # Build dataset of all cleaners for this job
    candidates = _build_candidate_frame(job)

    X_candidates = candidates[feature_columns].copy()
    candidates["predicted_compatibility"] = model.predict(X_candidates)
    candidates["predicted_compatibility"] = candidates["predicted_compatibility"].clip(0, 1)

    candidates["reason_tags"] = candidates.apply(lambda row: _reason_tags(row, job), axis=1)

    # Predict compatibility scores
    ranked = candidates.sort_values(
        by=["predicted_compatibility", "stars", "review_count"],
        ascending=[False, False, False],
    ).head(top_n)

    output_cols = [
        "business_id",
        "name",
        "city",
        "state",
        "stars",
        "review_count",
        "hourly_rate_est",
        "predicted_compatibility",
        "reason_tags",
        "categories",
    ]

    return ranked[output_cols].reset_index(drop=True), metrics


if __name__ == "__main__":
    model, metrics = train_model()
    print("Model trained.")
    print(metrics)