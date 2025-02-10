import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Penguin Body Mass Predictor")

st.image('./AI PENGUIN 2.png', caption='Picture made with AI', width=500)

st.markdown("""
    The dataset contains information about different species of penguins. It includes measurements such as bill length, bill depth, flipper length, body mass, and the sex of each penguin.

    ### Problem Statement
    The objective is to build a regression model to predict a penguin's body mass based on its physical characteristics (bill length, bill depth, flipper length, species, and sex). Understanding these relationships can be useful in biological and ecological studies of penguin populations.

    Here are a few important questions that you might seek to address:
    - Is there a relationship between the physical characteristics of penguins and their body mass?
    - How strong is the relationship between features like bill length, flipper length, and body mass?
    - Which features contribute the most to predicting a penguin's body mass?
    - How accurately can we estimate a penguin’s weight using regression?
    - Is the relationship between these features and body mass linear?    

    ### Data Description
    - **Species:** The species of the penguin (Adelie, Chinstrap, Gentoo).
    - **Bill Length (mm):** The length of the penguin’s bill (in mm).
    - **Bill Depth (mm):** The depth of the penguin’s bill (in mm).
    - **Flipper Length (mm):** The length of the penguin’s flipper (in mm).
    - **Sex:** The sex of the penguin (Male or Female).
    - **Body Mass (g):** The body mass of the penguin (in grams). This is the value we are predicting.
""")

st.image('./Image 3 Penguins.png', caption='The three different species', width=500)

st.sidebar.title("Additional Info")

w1 = st.sidebar.checkbox("Dataset", False)

@st.cache_data
def read_data():
    return pd.read_csv("penguins_cleaned.csv")[["species","bill_length_mm","flipper_length_mm", "body_mass_g", "sex"]]

df=read_data()

if w1:
	st.markdown(""" ### Dataset 
    Here is the dataset only with the values we decided to use
    """)
	st.dataframe(df,width=2000,height=500)

# For loading the trained model and scalers

# Load the trained model and scalers
@st.cache_data
def load_model():
    model = joblib.load("penguin_body_mass_model.pkl")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

# Load the trained model and scalers
linear_model, scaler_X, scaler_y = load_model()

st.markdown("## 📌 Predict Penguin Body Mass")

# User Input Fields for Feature Selection
bill_length = st.number_input("Bill Length (mm)", min_value=30.0, max_value=60.0, value=45.0, step=0.1)
flipper_length = st.number_input("Flipper Length (mm)", min_value=170.0, max_value=230.0, value=200.0, step=0.1)

# Species Selection with One-Hot Encoding
species = st.selectbox("Species", ["Adelie", "Chinstrap", "Gentoo"])
species_dict = {"Adelie": [1, 0, 0], "Chinstrap": [0, 1, 0], "Gentoo": [0, 0, 1]}
species_encoded = species_dict[species]

# Sex Selection with One-Hot Encoding
sex = st.selectbox("Sex", ["Female", "Male"])
sex_dict = {"Female": [1, 0], "Male": [0, 1]}
sex_encoded = sex_dict[sex]

# Prepare the input for scaling
input_features = np.array([[bill_length, flipper_length] + species_encoded + sex_encoded])

# Scale input using the saved scaler_X
input_scaled = scaler_X.transform(input_features)

# Make the prediction
predicted_scaled_mass = linear_model.predict(input_scaled)

# Convert the scaled prediction back to the original range
predicted_mass = scaler_y.inverse_transform(np.array(predicted_scaled_mass).reshape(-1, 1))[0][0]

# Display the Prediction
st.markdown("### 🐧 Estimated Penguin Body Mass:")
st.success(f"📌 Predicted Body Mass: **{predicted_mass:.2f} grams**")

# Display Performance Metrics
st.markdown("## 📊 Model Performance Metrics")

# Load the saved performance metrics
metrics = joblib.load("model_performance.pkl")

# Display model performance in Streamlit
st.markdown("### 📊 Model Performance Metrics (Test Set)")

st.write(f"📌 **Mean Squared Error (MSE):** `{metrics['MSE']:.4f}`")
st.write(f"📌 **Root Mean Squared Error (RMSE):** `{metrics['RMSE']:.4f}`")
st.write(f"📌 **Mean Absolute Error (MAE):** `{metrics['MAE']:.4f}`")
st.write(f"📌 **R-Squared (R²):** `{metrics['R2']:.4f}`")
