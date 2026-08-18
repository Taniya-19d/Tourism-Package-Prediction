import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Enter the customer details below to get a prediction.
""")

age = st.number_input("Age", min_value=18, max_value=65, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=20000.0)
city_tier = st.radio("City Tier", [1, 2, 3], horizontal=True)
num_trips = st.number_input("Average Annual Trips Taken", min_value=0)
preferred_star = st.radio("Preferred Hotel Rating by Customer", [1, 2, 3, 4, 5], horizontal=True)
num_visitors = st.number_input("Total Visitors with Customer", min_value=1, max_value=10, value=2)
num_children = st.number_input("Children Visiting with Customer(< 5 yrs)", min_value=0, max_value=10)
passport = st.selectbox("Does the customer has passport?", ["Yes", "No"])
own_car = st.selectbox("Does the customer owns any car?", ["Yes", "No"])
contact_type = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
pitch_duration = st.number_input("Pitch Duration (in Minutes)", min_value=0)
pitch_satisfaction = st.radio("Pitch Satisfaction Score", [1, 2, 3, 4, 5],  horizontal=True)
num_followups = st.selectbox("Number of Follow-ups Made", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=3)


input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "Occupation": occupation,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "CityTier": city_tier,
    "NumberOfTrips": num_trips,
    "PreferredPropertyStar": preferred_star,
    "NumberOfPersonVisiting": num_visitors,
    "NumberOfChildrenVisiting": num_children,
    "Passport": passport,
    "OwnCar": own_car,
    "TypeofContact": contact_type,
    "ProductPitched": product_pitched,
    "DurationOfPitch": pitch_duration,
    "PitchSatisfactionScore": pitch_satisfaction,
    "NumberOfFollowups": num_followups
}])

if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer will purchase the package" if prediction == 1 else "Customer will not purchase the package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
