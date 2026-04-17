import streamlit as st

st.set_page_config(page_title="SmartClean AI", layout="centered")

st.title("SmartClean AI")
st.caption("Intelligent cleaner matching for your home")

st.divider()

st.subheader("Tell us about your job")

with st.form("job_request"):
    col1, col2 = st.columns(2)

    with col1:
        cleaning_type = st.selectbox(
            "Cleaning type",
            ["Standard", "Deep Clean", "Move-Out", "Post-Construction"]
        )
        home_size = st.slider("Home size (sq ft)", 500, 5000, 1500, step=100)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)

    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=2)
        budget = st.slider("Budget ($/hr)", 20, 100, 40, step=5)
        requirements = st.multiselect(
            "Special requirements",
            ["Eco-friendly", "Pet-friendly", "Fast turnaround", "Detail-oriented"]
        )

    submitted = st.form_submit_button("Find Cleaners", use_container_width=True)

st.divider()

if submitted:
    st.subheader("Top Matches")
    st.info("Matching engine not connected yet.")
else:
    st.markdown("Fill out the form above and click **Find Cleaners** to get matched.")
