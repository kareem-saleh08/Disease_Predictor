"""
Heart Disease Prediction - Model Explainability

This script provides model interpretability using SHAP and LIME.
It helps understand how the model makes predictions and which features are most important.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from lime import lime_tabular
import json
import os

# Set style for plots
plt.style.use('seaborn')
sns.set_palette('viridis')

# Create necessary directories
os.makedirs('../reports/figures', exist_ok=True)

def load_data_and_model():
    """Load the dataset and the trained model."""
    # Load the dataset
    df = pd.read_csv('../data/heart.csv')
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Load the trained model
    model = joblib.load('../app/model.pkl')
    
    # Load feature names
    with open('../app/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    return X, y, model, feature_names

def get_preprocessor(model):
    """Extract the preprocessor from the model pipeline."""
    if hasattr(model, 'steps'):
        return model.named_steps['preprocessor']
    return None

def get_feature_names_after_preprocessing(preprocessor, feature_names):
    """Get feature names after preprocessing."""
    # Get feature names after one-hot encoding
    categorical_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        input_features=feature_names[1:8]  # Assuming first 8 are categorical
    )
    
    # Combine with numerical feature names
    numerical_features = feature_names[8:]  # Assuming last 5 are numerical
    all_features = list(categorical_features) + numerical_features
    
    return all_features

def shap_analysis(model, X, feature_names):
    """Perform SHAP analysis on the model."""
    print("Performing SHAP analysis...")
    
    # Get the preprocessor and model from the pipeline
    preprocessor = get_preprocessor(model)
    model = model.named_steps['classifier']
    
    # Transform the data
    X_processed = preprocessor.transform(X)
    
    # Get feature names after preprocessing
    processed_feature_names = get_feature_names_after_preprocessing(preprocessor, feature_names)
    
    # Create a SHAP explainer
    explainer = shap.Explainer(model, X_processed, feature_names=processed_feature_names)
    
    # Calculate SHAP values (use a subset for faster computation)
    X_sample = X_processed[:100]  # Use first 100 samples
    shap_values = explainer(X_sample)
    
    # Plot SHAP summary
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=processed_feature_names, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('../reports/figures/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot SHAP bar plot
    plt.figure(figsize=(12, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title('SHAP Feature Importance (Bar Plot)')
    plt.tight_layout()
    plt.savefig('../reports/figures/shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save SHAP values for specific examples
    for i in range(min(3, X.shape[0])):
        plt.figure(figsize=(10, 6))
        shap.plots.force(explainer.expected_value, shap_values[i, :], 
                         X_sample[i], feature_names=processed_feature_names,
                         matplotlib=True, show=False)
        plt.title(f'SHAP Force Plot - Example {i+1}')
        plt.tight_layout()
        plt.savefig(f'../reports/figures/shap_force_plot_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("SHAP analysis completed. Visualizations saved to ../reports/figures/")

def lime_analysis(model, X, y, feature_names, preprocessor):
    """Perform LIME analysis on the model."""
    print("\nPerforming LIME analysis...")
    
    # Get the class names
    class_names = ['No Disease', 'Disease']
    
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        preprocessor.transform(X).toarray(),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )
    
    # Explain predictions for a few examples
    for i in range(min(3, X.shape[0])):
        # Select an instance to explain
        instance = X.iloc[i].values.reshape(1, -1)
        
        # Get the explanation
        exp = explainer.explain_instance(
            preprocessor.transform(instance).toarray()[0],
            model.predict_proba,
            num_features=len(feature_names),
            top_labels=2
        )
        
        # Save the explanation as an image
        fig = exp.as_pyplot_figure()
        plt.title(f'LIME Explanation - Example {i+1} (True: {class_names[y.iloc[i]]})')
        plt.tight_layout()
        plt.savefig(f'../reports/figures/lime_explanation_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("LIME analysis completed. Visualizations saved to ../reports/figures/")

def main():
    # Load data and model
    print("Loading data and model...")
    X, y, model, feature_names = load_data_and_model()
    
    # Get the preprocessor
    preprocessor = get_preprocessor(model)
    
    # Get processed feature names
    processed_feature_names = get_feature_names_after_preprocessing(preprocessor, feature_names)
    
    # Perform SHAP analysis
    shap_analysis(model, X, feature_names)
    
    # Perform LIME analysis
    lime_analysis(model, X, y, processed_feature_names, preprocessor)
    
    print("\nModel explainability analysis completed!")

if __name__ == "__main__":
    main()
