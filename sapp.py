# %%
import streamlit as st
import joblib
import pandas as pd



# Load trained model

@st.cache_resource
def load_model():
    model = joblib.load("Linear_Regression.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    return model, preprocessor

model, preprocessor = load_model()




# Page Title

st.title("House Price Prediction")

st.write("Enter Your House Information To Predict Price")



# Create two columns

col1, col2 = st.columns(2)


# Inputs - Column 1


with col1:


    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=5,
        value=2
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=5,
        value=2
    )

    area_sqft = st.number_input(
        "Area_sqft",
        min_value=200,
        max_value=4000,
        value=1500
    )

    lot_size = st.number_input(
        "Lot_Size",
        min_value=400,
        max_value=7500,
        value=2500
    )

    floors = st.number_input(
        "Floors",
        min_value=1,
        max_value=3,
        value=2
    )

    house_age = st.number_input(
        "House_Age",
        min_value=0,
        max_value=40,
        value=19
    )

    garage_capacity = st.number_input(
        "Garage_Capacity",
        min_value=0,
        max_value=3,
        value=0
    )

    distance_to_city = st.number_input(
        "Distance_to_City",
        min_value=1,
        max_value=50,
        value=25
    )

    crime_rate = st.number_input(
        "Crime_Rate",
        min_value=0.0,
        max_value=15.0,
        value=7.0
    )

    school_rating = st.number_input(
        "School_Rating",
        min_value=1.0,
        max_value=5.0,
        value=2.5
    )


# Inputs column 2


with col2:

    city = st.selectbox(
        "City",
        options=("Thane", "Nashik", "Mumbai", "Nagpur", "Pune"),
        index=None,
        placeholder="Select City ?"
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        options=("Thane", "Nashik", "Mumbai", "Nagpur", "Pune"),
        index=None,
        placeholder="Select Neighborhood ?"
    )

    property_type = st.selectbox(
        "Property_Type",
        options=("Independent House", "Apartment", "Villa"),
        index=None,
        placeholder="Select Property Type ?"
    )

    has_garden = st.selectbox(
        "Has_Garden",
        options=(True, False),
        index=None,
        placeholder="Select Has Garden ?",
        format_func=lambda x: "Yes" if x else "No"

    )

    has_pool = st.selectbox(
        "Has_Pool",
        options=(True, False),
        index=None,
        placeholder="Select Has Pool ?",
        format_func=lambda x: "Yes" if x else "No"
    )

    condition = st.selectbox(
        "Condition",
        options=("Poor", "Fair", "Good", "Excellent"),
        index=None,
        placeholder="Select Condition ?"
    )

    heating_type = st.selectbox(
        "Heating_Type",
        options=("Gas", "Solar", "Electric"),
        index=None,
        placeholder="Select Heating Type ?"
    )

 

# Prediction Button

if st.button("Predict"):

    # Check mandatory fields

    if (  city is None
        or neighborhood is None
        or property_type is None
        or has_garden is None
        or has_pool is None
        or condition is None
        or heating_type is None
    ):

        st.error("Each Field is Mandatory")

    else:

        # Create DataFrame

        features = pd.DataFrame([{
           "City": city,
           "Neighborhood": neighborhood,
           "Property_Type": property_type,
           "Bedrooms": bedrooms,
           "Bathrooms": bathrooms,
           "Area_sqft": area_sqft,
           "Lot_Size": lot_size,
           "Floors": floors,
           "House_Age": house_age,
           "Garage_Capacity": garage_capacity,
           "Distance_to_City": distance_to_city,
           "Crime_Rate": crime_rate,
           "School_Rating": school_rating,
           "Has_Garden": has_garden,
           "Has_Pool": has_pool,
           "Condition": condition,
           "Heating_Type": heating_type


        }])

        # Prediction

        


        features = preprocessor.transform(features)

        prediction = model.predict(features)

        predicted_price = prediction[0]


        
        # Display Result
        

        st.success(
            f"Predicted House Price: ₹{predicted_price:,.2f}"
        )



