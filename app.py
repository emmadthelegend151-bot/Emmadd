import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Machine Learning Predictor",
    page_icon="🔮",
    layout="centered"
)

# 2. Load the Pickled Model Safely
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure it is in the same directory.")
        return None

model = load_model()

# 3. User Interface
st.title("🔮 Machine Learning Model Predictor")
st.write("Enter the required inputs below to generate a prediction.")

# --- CUSTOMIZE YOUR INPUTS HERE ---
# Replace these with the actual features your model expects
feature_1 = st.number_input("Feature 1 Name (e.g., Age)", min_value=0, max_value=100, value=25)
feature_2 = st.number_input("Feature 2 Name (e.g., Income)", min_value=0, value=50000)
feature_3 = st.selectbox("Feature 3 Name (e.g., Gender)", options=["Option A", "Option B"])

# Convert categorical inputs to numerical if your model requires it
f3_mapped = 1 if feature_3 == "Option A" else 0
# ----------------------------------

# 4. Prediction Logic
if st.button("Generate Prediction", type="primary"):
    if model is not None:
        # Prepare the input array matching the shape your model trained on
        input_data = np.array([[feature_1, feature_2, f3_mapped]])
        
        # Run prediction
        prediction = model.predict(input_data)
        
        # Display Results
        st.success("🎉 Prediction Generated Successfully!")
        st.metric(label="Predicted Value", value=f"{prediction[0]}")
    else:
        st.error("Model is not loaded. Cannot predict.")
