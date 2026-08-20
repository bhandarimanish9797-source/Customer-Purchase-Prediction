import streamlit as st
import pandas as pd
import joblib



# Load trained model
model = joblib.load("customer_purchase_model.pkl")

st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Customer Purchase Prediction")
st.write("Logistic Regression Model")

st.subheader("Enter Customer Details")

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

AnnualIncome = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=40000.0
)

DeviceType = st.selectbox(
    "Device Type",
    ["Mobile", "Desktop"]
)

TimeSpentOnSite = st.number_input(
    "Time Spent On Site",
    min_value=0.0,
    value=10.0
)

PagesViewed = st.number_input(
    "Pages Viewed",
    min_value=0,
    value=5
)

PreviousPurchases = st.number_input(
    "Previous Purchases",
    min_value=0,
    value=1
)

DiscountOffered = st.selectbox(
    "Discount Offered",
    ["Yes", "No"]
)

EmailOpened = st.selectbox(
    "Email Opened",
    ["Yes", "No"]
)

AdClicked = st.selectbox(
    "Ad Clicked",
    ["Yes", "No"]
)

MembershipType = st.selectbox(
    "Membership Type",
    ["Basic", "Premium", "No Membership"]
)

if st.button("Predict Purchase"):

    input_data = pd.DataFrame({
        "Gender": [Gender],
        "Age": [Age],
        "AnnualIncome": [AnnualIncome],
        "DeviceType": [DeviceType],
        "TimeSpentOnSite": [TimeSpentOnSite],
        "PagesViewed": [PagesViewed],
        "PreviousPurchases": [PreviousPurchases],
        "DiscountOffered": [DiscountOffered],
        "EmailOpened": [EmailOpened],
        "AdClicked": [AdClicked],
        "MembershipType": [MembershipType]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Purchase: YES")
    else:
        st.error("❌ Purchase: NO")