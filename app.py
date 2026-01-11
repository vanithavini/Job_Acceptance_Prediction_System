import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Job Acceptance Prediction System", layout="wide")

st.title("📊 Job Acceptance Prediction System")
st.markdown("Predict job acceptance and analyze placement insights")

# Paths
DATA_PATH = r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\job_acceptance_prediction_system\data\cleaned\job_acceptance_f_e_data.csv"
ARTIFACT_PATH = r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\job_acceptance_prediction_system\artifacts"

# Load Data & Artifacts
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_artifacts():
    with open(f"{ARTIFACT_PATH}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{ARTIFACT_PATH}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{ARTIFACT_PATH}/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open(f"{ARTIFACT_PATH}/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, scaler, label_encoders, feature_columns

df = load_data()
model, scaler, label_encoders, feature_columns = load_artifacts()

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Dashboard", "Predict Job Acceptance"])

# Dashboard
if menu == "Dashboard":
    st.header("📊 Placement Dashboard")

    # Clean status
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["status"] = df["status"].replace({
        "placed": "Placed",
        "yes": "Placed",
        "accepted": "Placed",
        "1": "Placed",
        "true": "Placed",
        "not placed": "Not Placed",
        "not_placed": "Not Placed",
        "rejected": "Not Placed",
        "no": "Not Placed",
        "0": "Not Placed",
        "false": "Not Placed"
    })

    total_candidates = len(df)
    placed_count = (df["status"] == "Placed").sum()
    not_placed_count = (df["status"] == "Not Placed").sum()
    placement_rate = (placed_count / total_candidates) * 100
    job_acceptance_rate = placement_rate  
    avg_interview = df["interview_score_avg"].mean()
    avg_skills_match = df["skills_match_percentage"].mean()
    offer_dropout_rate = (df["status"] == "Not Placed").mean() * 100
    high_risk_pct = (df["placement_probability_score"] < 0.4).mean() * 100

    # KPIs
    st.subheader("Key KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Candidates", total_candidates)
    c2.metric("Placement Rate (%)", f"{placement_rate:.2f}")
    c3.metric("Job Acceptance Rate (%)", f"{job_acceptance_rate:.2f}")
    c4.metric("Avg Interview Score", f"{avg_interview:.2f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Avg Skills Match (%)", f"{avg_skills_match:.2f}")
    c6.metric("Offer Dropout Rate (%)", f"{offer_dropout_rate:.2f}")
    c7.metric("High-Risk Candidates (%)", f"{high_risk_pct:.2f}")

    # Visualizations
    st.subheader("📈 Acceptance Rate by Company Tier")
    tier_rate = df.groupby("company_tier")["status"].apply(lambda x: (x == "Placed").mean()).reset_index(name="Acceptance Rate")
    st.bar_chart(tier_rate.set_index("company_tier"))

    st.subheader("🧠 Skills Match vs Placement Probability")
    skills_grp = df.groupby("skills_match_level")["placement_probability_score"].mean().reset_index()
    fig = px.line(skills_grp, x="skills_match_level", y="placement_probability_score", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# Prediction
if menu == "Predict Job Acceptance":
    st.header("🤖 Predict Job Acceptance")
    st.write("Enter candidate details below:")

    with st.form("prediction_form"):
        col1, col2, col3, col4 = st.columns(4)
        age = col1.number_input("Age", min_value=18, max_value=60, value=25, step=1)
        ssc = col2.number_input("SSC %", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        hsc = col3.number_input("HSC %", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        degree = col4.number_input("Degree %", min_value=0.0, max_value=100.0, value=70.0, step=0.1)

        col1, col2, col3 = st.columns(3)
        technical = col1.slider("Technical Score", 0, 100, 70)
        aptitude = col2.slider("Aptitude Score", 0, 100, 65)
        communication = col3.slider("Communication Score", 0, 100, 60)

        col1, col2, col3 = st.columns(3)
        skills_match = col1.number_input("Skills Match %", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        certifications = col2.number_input("Certifications Count", min_value=0, max_value=20, value=2, step=1)
        experience = col3.number_input("Years of Experience", min_value=0, max_value=40, value=2, step=1)

        col1, col2, col3 = st.columns(3)
        gender = col1.selectbox("Gender", ["Male", "Female"])
        internship = col2.selectbox("Internship Experience", ["Yes", "No"])
        company_tier = col3.selectbox("Company Tier", ["Tier 1", "Tier 2", "Tier 3"])

        submit = st.form_submit_button("Predict")

    if submit:
        # Derived features (same as training)
        interview_avg = (technical + aptitude + communication) / 3
        placement_prob = skills_match / 100
        academic_band = 2 if degree >= 75 else 1 if degree >= 60 else 0
        experience_category = 0 if experience < 2 else 1 if experience < 5 else 2
        interview_performance = 2 if interview_avg >= 75 else 1 if interview_avg >= 50 else 0
        if skills_match < 50:
            skills_match_level = "Low"
        elif skills_match < 75:
            skills_match_level = "Medium"
        else:
            skills_match_level = "High"

        # Input DataFrame
        input_data = {
            "age_years": age,
            "ssc_percentage": ssc,
            "hsc_percentage": hsc,
            "degree_percentage": degree,
            "technical_score": technical,
            "aptitude_score": aptitude,
            "communication_score": communication,
            "skills_match_percentage": skills_match,
            "certifications_count": certifications,
            "years_of_experience": experience,
            "interview_score_avg": interview_avg,
            "placement_probability_score": placement_prob,
            "academic_band": academic_band,
            "experience_category": experience_category,
            "interview_performance": interview_performance,
            "skills_match_level": skills_match_level,
            "gender": gender,
            "internship_experience": internship,
            "company_tier": company_tier
        }

        input_df = pd.DataFrame([input_data])

        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in input_df:
                val = input_df.at[0, col]
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    input_df[col] = 0

        # Ensure numeric columns
        numeric_cols = [col for col in feature_columns if col not in label_encoders]
        for col in numeric_cols:
            if col not in input_df:
                input_df[col] = 0
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

        # Apply scaler
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Reorder columns
        input_df = input_df[feature_columns]

        # Prediction
        pred = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0][1]
        result = "Placed ✅" if pred == 1 else "Not Placed ❌"

        st.success(f"Prediction: {result}")
        st.info(f"Acceptance Probability: {pred_prob*100:.2f}%")
