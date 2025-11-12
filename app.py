"""
Heart Disease Prediction - Streamlit App

This script creates an interactive web application for heart disease prediction
using the trained model and provides explanations using SHAP values.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model and feature names
@st.cache_resource
def load_model():
    """Load the trained model and feature names."""
    model_path = Path(__file__).parent / 'model.pkl'
    feature_names_path = Path(__file__).parent / 'feature_names.json'
    
    model = joblib.load(model_path)
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    return model, feature_names

# Get the preprocessor and model from the pipeline
def get_model_components(model):
    """Extract the preprocessor and model from the pipeline."""
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    return preprocessor, classifier

# Create a function to get feature importance
def get_feature_importance(model, feature_names):
    """Get feature importance from the model."""
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    # Create a DataFrame with feature names and importance scores
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return importance_df

# Create a function to generate SHAP explanations
def generate_shap_explanation(model, input_data, feature_names):
    """Generate SHAP explanation for the prediction."""
    # Extract the preprocessor and model
    preprocessor, classifier = get_model_components(model)
    
    # Transform the input data
    input_processed = preprocessor.transform(input_data)
    
    # Create a SHAP explainer
    explainer = shap.Explainer(classifier, input_processed, feature_names=feature_names)
    
    # Calculate SHAP values
    shap_values = explainer(input_processed)
    
    return explainer, shap_values

# Create the main app
def main():
    """Main function to run the Streamlit app."""
    # Load the model and feature names
    model, feature_names = load_model()
    
    # Set app title and description
    st.title("❤️ Heart Disease Prediction")
    st.markdown("""
    This app predicts the likelihood of heart disease based on patient data.
    The model uses machine learning to analyze various health indicators.
    """)
    
    # Create a sidebar for user input
    st.sidebar.header("Patient Information")
    
    # Create input fields for each feature
    st.sidebar.subheader("Demographic Information")
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.radio("Sex", ["Male", "Female"])
    
    st.sidebar.subheader("Medical History")
    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.sidebar.selectbox(
        "Resting Electrocardiographic Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    
    st.sidebar.subheader("Exercise Test Results")
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.radio("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0, 0.1)
    slope = st.sidebar.selectbox(
        "Slope of the Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )
    
    st.sidebar.subheader("Other Factors")
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
    thal = st.sidebar.selectbox(
        "Thalassemia",
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )
    
    # Convert inputs to model format
    def preprocess_input():
        """Convert user inputs to the format expected by the model."""
        input_data = {
            'age': age,
            'sex': 1 if sex == "Male" else 0,
            'chest_pain': ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
            'resting_bp': trestbps,
            'cholesterol': chol,
            'fasting_blood_sugar': 1 if fbs == "Yes" else 0,
            'rest_ecg': ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
            'max_heart_rate': thalach,
            'exercise_induced_angina': 1 if exang == "Yes" else 0,
            'st_depression': oldpeak,
            'slope': ["Upsloping", "Flat", "Downsloping"].index(slope),
            'num_major_vessels': ca,
            'thalassemia': ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
        }
        return pd.DataFrame([input_data])
    
    # Create a button to make predictions
    if st.sidebar.button('Predict Heart Disease'):
        # Preprocess input data
        input_data = preprocess_input()
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display prediction
        st.subheader("Prediction Result")
        
        # Create columns for the result
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", "Disease" if prediction == 1 else "No Disease")
        
        with col2:
            st.metric("Probability", f"{probability * 100:.1f}%")
        
        # Add a gauge chart for probability
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh([""], [probability * 100], height=0.5, color="#FF4B4B" if probability > 0.5 else "#4CAF50")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_title('Disease Probability')
        st.pyplot(fig)
        
        # Show feature importance
        st.subheader("Feature Importance")
        
        # Get feature importance
        preprocessor, classifier = get_model_components(model)
        input_processed = preprocessor.transform(input_data)
        
        # Get feature names after preprocessing
        try:
            # For one-hot encoded features
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
                input_features=feature_names[1:8]
            )
            num_features = feature_names[8:]
            all_features = list(cat_features) + num_features
        except:
            all_features = feature_names
        
        # Generate SHAP explanation
        explainer, shap_values = generate_shap_explanation(
            model, input_data, all_features
        )
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["SHAP Summary", "SHAP Force Plot"])
        
        with tab1:
            # Plot SHAP summary
            st.markdown("### How each feature contributes to the prediction")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_values[0], 
                input_processed, 
                feature_names=all_features,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig)
        
        with tab2:
            # Plot SHAP force plot
            st.markdown("### How each feature contributes to this specific prediction")
            fig, ax = plt.subplots(figsize=(12, 4))
            shap.plots.force(
                explainer.expected_value[1],
                shap_values[0, :, 1],
                input_processed[0],
                feature_names=all_features,
                matplotlib=True,
                show=False
            )
            st.pyplot(fig)
        
        # Add interpretation
        st.subheader("Interpretation")
        if prediction == 1:
            st.warning("""
            The model predicts a **high probability** of heart disease. 
            Please consult with a healthcare professional for further evaluation.
            """)
        else:
            st.success("""
            The model predicts a **low probability** of heart disease. 
            However, this is not a medical diagnosis. Please consult with a healthcare 
            professional for a comprehensive evaluation.
            """)
    
    # Add information about the model
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### About
    This app uses a machine learning model to predict the likelihood of heart disease 
    based on patient data. The model was trained on the UCI Heart Disease dataset.
    
    **Note:** This is for educational purposes only and not a substitute for 
    professional medical advice.
    """)

if __name__ == "__main__":
    main()
