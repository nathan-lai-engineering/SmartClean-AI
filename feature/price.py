# feature: for the customer to see the estimated price for their house

def estimate_price(
    square_feet: int,
    job_type: str,
    bedrooms: int = 3,
    bathrooms: int = 2,
    house_age: int = 20,
    extras: dict = None
):
    """
    Estimate total cleaning price based on multiple factors.

    Input:
    - square_feet: size of the home
    - job_type: type of cleaning (standard, deep_clean, move_out, post_construction)
    - bedrooms: number of bedrooms
    - bathrooms: number of bathrooms
    - house_age: older homes may require more effort
    - extras: optional dictionary of special requirements (bool flags)

    Output:
    - Estimated total price (USD)
    """

    # Base rate per square foot
    # base cleaning cost per sq ft
    base_rate = 0.08  

    # Job type multiplier
    job_multipliers = {
        "standard": 1.0,
        "deep_clean": 1.5,
        "move_out": 1.3,
        "post_construction": 1.6
    }
    multiplier = job_multipliers.get(job_type, 1.0)

    # Room adjustments 
    bedroom_cost = bedrooms * 15
    bathroom_cost = bathrooms * 25

    # House age adjustment
    if house_age > 50:
        age_factor = 1.2
    elif house_age > 20:
        age_factor = 1.1
    else:
        age_factor = 1.0

    # Extras / special requirements
    extra_cost = 0
    if extras:
        if extras.get("pet_friendly"):
            extra_cost += 20
        if extras.get("eco_friendly"):
            extra_cost += 15
        if extras.get("window_cleaning"):
            extra_cost += 30
        if extras.get("fast_turnaround"):
            extra_cost += 25
        if extras.get("detail_oriented"):
            extra_cost += 20

    # Base calculation
    base_price = square_feet * base_rate

    # Final price calculation
    estimated_price = (base_price * multiplier * age_factor) \
                      + bedroom_cost \
                      + bathroom_cost \
                      + extra_cost

    return round(estimated_price, 2)


if __name__ == "__main__":
    square_feet = int(input("Enter square feet: "))
    job_type = input("Enter job type: ")

    price = estimate_price(square_feet, job_type)
    print("Estimated price:", price)