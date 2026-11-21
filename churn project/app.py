
import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer will leave the bank.")

# --------------------------------
# 2. Load Pre-trained Model
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("churn_model.pkl not found. Place it in the same folder as app.py.")
    st.stop()

# --------------------------------
# 3. Customer Input Form
# --------------------------------

st.subheader("Enter Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        CreditScore = st.number_input("Credit Score", 300, 900, 650)
        Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Age = st.number_input("Age", 18, 100, 40)
        Tenure = st.number_input("Tenure", 0, 10, 5)

    with col2:
        Balance = st.number_input("Balance", min_value=0.0, value=50000.0)
        NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])
        HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"])
        IsActiveMember = st.selectbox("Is Active Member?", ["Yes", "No"])
        EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=75000.0)

    submit = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

# --------------------------------
# 4. Prediction
# --------------------------------

if submit:

    customer_data = pd.DataFrame({
        "CreditScore": [CreditScore],
        "Geography": [Geography],
        "Gender": [Gender],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [1 if HasCrCard == "Yes" else 0],
        "IsActiveMember": [1 if IsActiveMember == "Yes" else 0],
        "EstimatedSalary": [EstimatedSalary]
    })

    prediction = model.predict(customer_data)[0]
    probability = model.predict_proba(customer_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to EXIT (Churn)")
    else:
        st.success("✅ Customer is likely to STAY")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

# --------------------------------
# 5. Footer
# --------------------------------

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0E1117;
        color: #A0A0A0;
        text-align: center;
        padding: 12px;
        font-size: 14px;
        border-top: 1px solid #30363D;
        z-index: 999;
    }

    .footer span {
        color: #FFFFFF;
        font-weight: bold;
    }
    </style>

    <div class="footer">
        Developed by <span>Kanishk Jain</span>
    </div>
    """,
    unsafe_allow_html=True
)
