import re
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from models.feature_extractor import extract_features, extract_features_groq

st.set_page_config(page_title="SmartClean AI", layout="centered")

st.title("SmartClean AI")
st.caption("Intelligent cleaner matching for your home")

st.divider()

st.subheader("Tell us about your job")

tab_form, tab_text = st.tabs(["Structured Form", "Free Text"])

job = None

# structured form to help guide selections
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

# --- Results ---
st.divider()

if job:
    st.subheader("Top Matches")
    st.info("Matching engine not connected yet.")

    with st.expander("Parsed job features (debug)"):
        st.json(job)
else:
    st.markdown("Fill out the form or describe your job above, then click **Find Cleaners**.")
