import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_JOBS = 200
CLEANERS_PER_JOB = 15


def safe_clip(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def main():
    np.random.seed(RANDOM_SEED)

    ames = pd.read_csv("./data/AmesHousing.csv")
    cleaners = pd.read_csv("./data/cleaner_profiles.csv")

    # Keep only rows with the housing fields we need
    required_cols = ["Gr Liv Area", "Bedroom AbvGr", "Full Bath", "Half Bath", "Year Built"]
    ames = ames.dropna(subset=required_cols).copy()

    # Sample homes for synthetic jobs
    ames_sample = ames.sample(min(N_JOBS, len(ames)), random_state=RANDOM_SEED).copy()

    rows = []

    for job_id, (_, home) in enumerate(ames_sample.iterrows(), start=1):
        gr_liv_area = float(home["Gr Liv Area"])
        bedroom_count = int(home["Bedroom AbvGr"])
        full_bath = int(home["Full Bath"])
        half_bath = int(home["Half Bath"])
        year_built = int(home["Year Built"])

        house_age = 2026 - year_built

        # Estimated cleaning duration
        estimated_hours = (
            1.5
            + 0.0015 * gr_liv_area
            + 0.4 * bedroom_count
            + 0.6 * full_bath
            + 0.3 * half_bath
            + 0.003 * house_age
        )

        # Synthetic job requirements
        needs_deep_clean = np.random.binomial(1, 0.30)
        needs_move_out = np.random.binomial(1, 0.15)
        needs_pet_friendly = np.random.binomial(1, 0.20)
        needs_fast_turnaround = np.random.binomial(1, 0.25)
        needs_detail_oriented = np.random.binomial(1, 0.35)
        needs_eco_friendly = np.random.binomial(1, 0.15)

        if needs_move_out:
            job_type = "move_out"
        elif needs_deep_clean:
            job_type = "deep_clean"
        elif needs_fast_turnaround:
            job_type = "fast_turnaround"
        else:
            job_type = "standard"

        # Synthetic budget based around estimated labor
        target_budget_per_hour = np.random.normal(35, 8)
        target_budget_per_hour = max(18, target_budget_per_hour)

        sampled_cleaners = cleaners.sample(
            min(CLEANERS_PER_JOB, len(cleaners)),
            random_state=job_id
        )

        for _, cleaner in sampled_cleaners.iterrows():
            cleaner_rating_scaled = cleaner["cleaner_rating_scaled"]
            review_count_scaled = cleaner["review_count_scaled"]
            hourly_rate_est = float(cleaner["hourly_rate_est"])

            specialization_fit = (
                needs_deep_clean * cleaner["deep_clean"]
                + needs_move_out * cleaner["move_out"]
                + needs_pet_friendly * cleaner["pet_friendly"]
                + needs_fast_turnaround * cleaner["fast_turnaround"]
                + needs_detail_oriented * cleaner["detail_oriented"]
                + needs_eco_friendly * cleaner["eco_friendly"]
            ) / 6.0

            professionalism_fit = (
                cleaner["reliable"]
                + cleaner["communicative"]
                + cleaner["experienced"]
            ) / 3.0

            price_fit = 1 - abs(target_budget_per_hour - hourly_rate_est) / target_budget_per_hour
            price_fit = safe_clip(price_fit)

            compatibility_score = (
                0.40 * specialization_fit
                + 0.20 * cleaner_rating_scaled
                + 0.10 * review_count_scaled
                + 0.15 * price_fit
                + 0.15 * professionalism_fit
            )

            compatibility_score += np.random.normal(0, 0.05)
            compatibility_score = safe_clip(compatibility_score)

            rows.append({
                "job_id": job_id,
                "business_id": cleaner["business_id"],
                "job_type": job_type,
                "gr_liv_area": gr_liv_area,
                "bedroom_abvgr": bedroom_count,
                "full_bath": full_bath,
                "half_bath": half_bath,
                "house_age": house_age,
                "estimated_hours": estimated_hours,
                "target_budget_per_hour": target_budget_per_hour,
                "needs_deep_clean": needs_deep_clean,
                "needs_move_out": needs_move_out,
                "needs_pet_friendly": needs_pet_friendly,
                "needs_fast_turnaround": needs_fast_turnaround,
                "needs_detail_oriented": needs_detail_oriented,
                "needs_eco_friendly": needs_eco_friendly,
                "cleaner_rating": cleaner["stars"],
                "cleaner_review_count": cleaner["review_count"],
                "hourly_rate_est": hourly_rate_est,
                "deep_clean": cleaner["deep_clean"],
                "move_out": cleaner["move_out"],
                "pet_friendly": cleaner["pet_friendly"],
                "fast_turnaround": cleaner["fast_turnaround"],
                "detail_oriented": cleaner["detail_oriented"],
                "eco_friendly": cleaner["eco_friendly"],
                "reliable": cleaner["reliable"],
                "communicative": cleaner["communicative"],
                "experienced": cleaner["experienced"],
                "specialization_fit": specialization_fit,
                "professionalism_fit": professionalism_fit,
                "price_fit": price_fit,
                "compatibility_score": compatibility_score
            })

    training = pd.DataFrame(rows)
    training.to_csv("./data/training_data.csv", index=False)

    print("Saved data/training_data.csv")
    print(f"Rows: {len(training)}")
    print(training.head())
    print("\nCompatibility score summary:")
    print(training["compatibility_score"].describe())


if __name__ == "__main__":
    main()