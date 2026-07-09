"""
Streamlit web app — Loan Default Prediction
Loads the trained model bundle (loan_default_model.pkl) produced by
model2_loan_default_prediction.py and lets a user enter one applicant's
info to get a default-risk prediction.

Run with:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# 1. LOAD MODEL BUNDLE (once, cached — not on every click)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_bundle():
    return joblib.load('loan_default_model.pkl')

bundle = load_model_bundle()
model = bundle['model']
scaler = bundle['scaler']
feature_columns = bundle['feature_columns']
model_name = bundle['model_name']

# ─────────────────────────────────────────────────────────────────
# 2. PAGE CONFIG & TITLE
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦")
st.title("🏦 Loan Default Prediction")
st.caption(f"Model: {model_name}")

# ─────────────────────────────────────────────────────────────────
# 3. INPUT FORM
# ─────────────────────────────────────────────────────────────────
with st.form("loan_form"):
    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        person_emp_exp = st.number_input("Years of Employment Experience", min_value=0, max_value=60, value=5)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)

    with col2:
        person_gender = st.selectbox("Gender", ["female", "male"])
        person_education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Predict")

# ─────────────────────────────────────────────────────────────────
# 4. PREDICTION (runs only after the button is clicked)
# ─────────────────────────────────────────────────────────────────
if submitted:
    raw_input = pd.DataFrame([{
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'credit_score': credit_score,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file,
    }])

    raw_input['Income_per_Emp_Year'] = raw_input['person_income'] / (raw_input['person_emp_exp'] + 1)
    raw_input['Credit_History_per_Age'] = raw_input['cb_person_cred_hist_length'] / (raw_input['person_age'] + 1)

    cat_cols = ['person_gender', 'person_education', 'person_home_ownership',
                'loan_intent', 'previous_loan_defaults_on_file']
    encoded_input = pd.get_dummies(raw_input, columns=cat_cols)

    encoded_input = encoded_input.reindex(columns=feature_columns, fill_value=0)

    scaled_input = scaler.transform(encoded_input)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0, 1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High risk of DEFAULT — probability {probability:.1%}")
    else:
        st.success(f"✅ Likely to REPAY ON TIME — default probability only {probability:.1%}")

    st.progress(float(probability))
