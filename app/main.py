import re
import sys
import os
from urllib.parse import quote_plus
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from models.feature_extractor import extract_features, extract_features_groq
from models.matching_model import rank_cleaners

st.set_page_config(page_title="SmartClean AI", layout="centered")

st.title("SmartClean AI")
st.caption("Intelligent cleaner matching for your home")

st.divider()

st.subheader("Tell us about your job")

tab_text, tab_form = st.tabs(["Free Text", "Structured Form"])

job = None

# --- Structured Form ---
with tab_form:
    with st.form("job_form"):
        col1, col2 = st.columns(2)

        with col1:
            cleaning_type = st.selectbox(
                "Cleaning type",
                ["Standard", "Deep Clean", "Move-Out", "Post-Construction"]
            )
            home_size   = st.slider("Home size (sq ft)", 500, 5000, 1500, step=100)
            bedrooms    = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            full_bath   = st.number_input("Bathrooms", min_value=1, max_value=8, value=2)

        with col2:
            house_age   = st.number_input("Approx. home age (years)", min_value=0, max_value=150, value=20)
            budget      = st.slider("Budget ($/hr)", 20, 100, 40, step=5)
            requirements = st.multiselect(
                "Special requirements",
                ["Eco-friendly", "Pet-friendly", "Fast turnaround", "Detail-oriented",
                 "Window cleaning", "Office / commercial"]
            )

        submitted_form = st.form_submit_button("Find Cleaners", use_container_width=True)

    if submitted_form:
        req = [re.sub(r'[\s\-/]+', '_', r.lower()) for r in requirements]

        job = {
            "job_type":               ("move_out" if cleaning_type == "Move-Out"
                                       else "deep_clean" if cleaning_type == "Deep Clean"
                                       else "post_construction" if cleaning_type == "Post-Construction"
                                       else "standard"),
            "gr_liv_area":            home_size,
            "bedroom_abvgr":          bedrooms,
            "full_bath":              full_bath,
            "house_age":              house_age,
            "estimated_hours":        (1.5 + 0.0015*home_size + 0.4*bedrooms
                                       + 0.6*full_bath + 0.003*house_age),
            "target_budget_per_hour": float(budget),
            "deep_clean":        int(cleaning_type == "Deep Clean"),
            "move_out":          int(cleaning_type == "Move-Out"),
            "post_construction": int(cleaning_type == "Post-Construction"),
            "pet_friendly":      int("pet_friendly" in req),
            "fast_turnaround":   int("fast_turnaround" in req),
            "detail_oriented":   int("detail_oriented" in req),
            "eco_friendly":      int("eco_friendly" in req),
            "window_cleaning":   int("window_cleaning" in req),
            "office_commercial": int("office_commercial" in req),
        }

# --- Free Text ---
with tab_text:
    st.caption("Prefer to fill out a form instead? Switch to the **Structured Form** tab right above this message.")
    user_text = st.text_area(
        "Describe your cleaning job",
        placeholder="e.g. I need a deep clean for a 3 bed 2 bath house, about 1800 sq ft. Budget around $45/hr. We have pets.",
        height=120
    )
    use_groq = st.toggle("Use AI parsing (Groq)", value=False,
                         help="Uses Groq LLM for better understanding of complex requests. Requires GROQ_API_KEY.")
    submitted_text = st.button("Find Cleaners", key="text_submit", use_container_width=True)

    if submitted_text:
        if not user_text.strip():
            st.warning("Please enter a description.")
        elif use_groq:
            with st.spinner("Parsing with AI..."):
                try:
                    job = extract_features_groq(user_text)
                except Exception as e:
                    st.warning(f"Groq parsing failed ({e}), falling back to keyword extraction.")
                    job = extract_features(user_text)
        else:
            job = extract_features(user_text)

# --- Constants ---
TAG_LABELS = {
    "deep_clean":        "Deep Clean",
    "move_out":          "Move-Out",
    "post_construction": "Post-Construction",
    "eco_friendly":      "Eco-Friendly",
    "pet_friendly":      "Pet-Friendly",
    "window_cleaning":   "Window Cleaning",
    "office_commercial": "Office / Commercial",
    "detail_oriented":   "Detail-Oriented",
    "fast_turnaround":   "Fast Turnaround",
    "reliable":          "Reliable",
    "communicative":     "Communicative",
    "experienced":       "Experienced",
}

JOB_TYPE_LABELS = {
    "standard":          "Standard Clean",
    "deep_clean":        "Deep Clean",
    "move_out":          "Move-Out",
    "post_construction": "Post-Construction",
    "fast_turnaround":   "Fast Turnaround",
}

CAPABILITY_TAGS = [
    "deep_clean", "move_out", "post_construction", "eco_friendly",
    "pet_friendly", "window_cleaning", "office_commercial",
    "detail_oriented", "fast_turnaround",
]


def render_score_badge(score):
    if score >= 0.65:
        color, bg = "#2d6a4f", "#d8f3dc"   # green — strong match
    elif score >= 0.40:
        color, bg = "#4a4a4a", "#e0e0e0"   # gray — mid match
    else:
        color, bg = "#b5451b", "#fde8df"   # red-orange — weak match
    return (
        f'<span style="background:{bg};color:{color};padding:4px 12px;'
        f'border-radius:12px;font-weight:700;font-size:1rem;">{score:.0%}</span>'
    )


def render_tag_badge(label, matched):
    if matched:
        style = "background:#d8f3dc;color:#2d6a4f;"   # green — tag met
    else:
        style = "background:#f0f0f0;color:#999;"       # gray — tag not met
    return (
        f'<span style="{style}padding:3px 9px;border-radius:10px;'
        f'font-size:0.8rem;margin-right:4px;">{label}</span>'
    )


# --- Results ---
st.divider()

if job:
    job_type_label = JOB_TYPE_LABELS.get(job.get("job_type", "standard"), "Standard Clean")
    active_tags    = [TAG_LABELS[t] for t in TAG_LABELS if job.get(t, 0) == 1]
    hours          = job.get("estimated_hours", 0)
    budget_val     = job.get("target_budget_per_hour", 45.0)

    # Exclude the tag that matches the job type to avoid repeating it
    job_type_tag = job.get("job_type", "standard")
    extra_tags = [TAG_LABELS[t] for t in TAG_LABELS
                  if job.get(t, 0) == 1 and t != job_type_tag]

    # Input summary
    summary_parts = [
        f"**{job_type_label}**",
        f"{int(job.get('bedroom_abvgr', 3))} bed · {int(job.get('full_bath', 2))} bath",
        f"{int(job.get('gr_liv_area', 1500))} sq ft",
        f"~{hours:.1f} hrs estimated",
        f"${budget_val:.0f}/hr budget",
    ]
    if extra_tags:
        summary_parts.append(" · ".join(extra_tags))
    st.info("  ·  ".join(summary_parts))

    requested = [t for t in CAPABILITY_TAGS if job.get(t, 0) == 1]
    if not requested:
        st.warning(
            "💡 **Tip:** Adding details like 'pet-friendly', 'eco-friendly', 'deep clean', "
            "or 'fast turnaround' helps us find a better match for you.",
            icon=None,
        )

    matches, metrics = rank_cleaners(job, top_n=5)

    if matches.empty:
        st.warning(
            "No cleaners matched your requirements. "
            "Try broadening your job description or removing some special requirements."
        )
    else:
        st.subheader(f"Top matches for your {job_type_label.lower()}")

        for i, row in matches.iterrows():
            score      = row["predicted_compatibility"]
            rate       = row["hourly_rate_est"]
            budget_diff = rate - budget_val

            with st.container(border=True):
                col_name, col_badge = st.columns([4, 1])

                with col_name:
                    st.markdown(f"#### {i + 1}. {row['name']}")
                    st.caption(f"{row['city']}, {row['state']}")

                with col_badge:
                    st.markdown(render_score_badge(score), unsafe_allow_html=True)

                col_stars, col_rate = st.columns([2, 1])

                with col_stars:
                    filled  = "★" * int(round(row["stars"]))
                    empty   = "☆" * (5 - int(round(row["stars"])))
                    st.markdown(f"{filled}{empty} **{row['stars']}** ({int(row['review_count'])} reviews)")

                with col_rate:
                    if abs(budget_diff) <= 8:
                        rate_bg, rate_color = "#2d6a4f", "white"
                    elif budget_diff > 0:
                        rate_bg, rate_color = "#b5451b", "white"
                    else:
                        rate_bg, rate_color = "#5a5a5a", "white"
                    rate_html = (
                        f'<span style="background:{rate_bg};color:{rate_color};'
                        f'padding:5px 12px;border-radius:10px;font-weight:700;'
                        f'font-size:1.1rem;">${rate:.0f}/hr</span>'
                    )
                    st.markdown(rate_html, unsafe_allow_html=True)

                if requested:
                    badges = "".join(
                        render_tag_badge(TAG_LABELS[t], row.get(t, 0) == 1)
                        for t in requested
                    )
                    st.markdown(badges, unsafe_allow_html=True)

                yelp_url = (
                    "https://www.yelp.com/search?find_desc="
                    + quote_plus(row["name"])
                    + "&find_loc="
                    + quote_plus(f"{row['city']}, {row['state']}")
                )
                st.markdown(f"[Search on Yelp]({yelp_url})", unsafe_allow_html=False)

                with st.expander("Categories"):
                    st.write(row["categories"])

    with st.expander("Model metrics"):
        st.json(metrics)

    with st.expander("Parsed job features (debug)"):
        st.json(job)

else:
    st.markdown("Fill out the form or describe your job above, then click **Find Cleaners**.")
