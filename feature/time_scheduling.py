# feature for checking the schedule to assign the best available cleaners

from datetime import datetime


TIME_SLOTS = [
    "08:00", "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00", "15:00",
    "16:00", "17:00", "18:00"
]


def recommend_time_slots(
    preferred_times: list,
    urgency: str,
    cleaner_available_times: list,
    estimated_hours: float,
    preferred_period: str = "any"
):
    """
    Recommend the best cleaning time slots based on:
    - customer preferred times
    - cleaner availability
    - urgency level
    - estimated job duration
    - preferred time of day

    Output:
    - top 3 recommended time scheduling and why
    """

    recommendations = []

    for slot in TIME_SLOTS:
        score = 0
        reasons = []

        hour = int(slot.split(":")[0])

        # Cleaner must be available
        if slot not in cleaner_available_times:
            continue

        score += 10
        reasons.append("Cleaner is available")

        # Customer preferred time match
        if slot in preferred_times:
            score += 8
            reasons.append("Matches customer preferred time")

        # Urgency
        if urgency == "high":
            if hour < 12:
                score += 6
                reasons.append("Earlier time is better for urgent jobs")
            elif hour < 16:
                score += 3
                reasons.append("Still reasonable for urgent job")

        elif urgency == "medium":
            if 10 <= hour <= 15:
                score += 4
                reasons.append("Balanced time for medium urgency")

        elif urgency == "low":
            if hour >= 14:
                score += 4
                reasons.append("Later time works for low urgency")

        # Preferred time
        if preferred_period == "morning" and hour < 12:
            score += 5
            reasons.append("Fits preferred morning schedule")
        elif preferred_period == "afternoon" and 12 <= hour < 17:
            score += 5
            reasons.append("Fits preferred afternoon schedule")
        elif preferred_period == "evening" and hour >= 17:
            score += 5
            reasons.append("Fits preferred evening schedule")
        elif preferred_period == "any":
            score += 2
            reasons.append("Flexible schedule")

        # Estimated job length
        if estimated_hours >= 4 and hour >= 15:
            score -= 5
            reasons.append("Long job may be difficult to finish late")
        elif estimated_hours <= 2:
            score += 2
            reasons.append("Short job is easier to schedule")

        recommendations.append({
            "time_slot": slot,
            "score": score,
            "reasons": reasons
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)

    return recommendations[:3]


if __name__ == "__main__":
    print("Scheduling Recommender")

    preferred_input = input("Enter preferred times separated by commas (example: 08:00,10:00,14:00): ")
    preferred_times = [time.strip() for time in preferred_input.split(",") if time.strip()]

    available_input = input("Enter cleaner available times separated by commas (example: 08:00,12:00,16:00): ")
    cleaner_available_times = [time.strip() for time in available_input.split(",") if time.strip()]

    urgency = input("Enter urgency level (low, medium, high): ").lower().strip()

    estimated_hours = float(input("Enter estimated job length in hours: "))

    preferred_period = input("Preferred period (morning, afternoon, evening, any): ").lower().strip()

    results = recommend_time_slots(
        preferred_times=preferred_times,
        urgency=urgency,
        cleaner_available_times=cleaner_available_times,
        estimated_hours=estimated_hours,
        preferred_period=preferred_period
    )

    print("\nRecommended Time Scheduling:")

    if not results:
        print("No available time slots matched the cleaner availability.")
    else:
        for index, result in enumerate(results, start=1):
            print(f"\n{index}. {result['time_slot']}")
            print(f"Score: {result['score']}")
            print("Reasons:")
            for reason in result["reasons"]:
                print(f"- {reason}")