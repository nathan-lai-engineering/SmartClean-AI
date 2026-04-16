import pandas as pd


def main():
    businesses = pd.read_csv("../data/yelp_businesses_clean.csv")
    tags = pd.read_csv("../data/demo_cleaner_tags.csv")
    wages = pd.read_csv("../data/state_wages.csv")

    # Keep only the business columns we actually want
    business_cols = [
        "business_id",
        "name",
        "city",
        "state",
        "stars",
        "review_count",
        "categories",
        "is_open"
    ]
    businesses = businesses[business_cols].copy()

    # Tag columns from your actual file
    tag_cols = [
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
        "experienced"
    ]

    # Merge businesses with tag data
    profiles = businesses.merge(
        tags[["business_id"] + tag_cols],
        on="business_id",
        how="left"
    )

    # Merge state wages using Yelp state abbreviation -> wage table state_abbr
    profiles = profiles.merge(
        wages[["state_abbr", "avg_hourly_wage"]],
        left_on="state",
        right_on="state_abbr",
        how="left"
    )

    # Rename for clarity
    profiles = profiles.rename(columns={"avg_hourly_wage": "hourly_rate_est"})

    # Fill missing wage values with median
    profiles["hourly_rate_est"] = profiles["hourly_rate_est"].fillna(
        profiles["hourly_rate_est"].median()
    )

    # Fill missing tag values with 0
    for col in tag_cols:
        profiles[col] = profiles[col].fillna(0).astype(int)

    # Derived helper columns
    profiles["cleaner_rating_scaled"] = profiles["stars"] / 5.0
    profiles["review_count_scaled"] = (profiles["review_count"] / 100.0).clip(upper=1.0)

    # Optional: keep only open businesses
    profiles = profiles[profiles["is_open"] == 1].copy()

    # Drop redundant wage merge key
    profiles = profiles.drop(columns=["state_abbr"])

    profiles.to_csv("../data/cleaner_profiles.csv", index=False)
    print("Saved data/cleaner_profiles.csv")
    print(f"Rows: {len(profiles)}")
    print(profiles.head())


if __name__ == "__main__":
    main()