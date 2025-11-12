"""
Heart Disease Prediction - Modeling

This script trains and evaluates machine learning models for heart disease prediction.
It includes data preprocessing, model training, hyperparameter tuning, and evaluation.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
)
import joblib
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
plt.style.use('seaborn')
sns.set_palette('viridis')

# Create necessary directories
os.makedirs('../reports/figures', exist_ok=True)
os.makedirs('../app', exist_ok=True)

def load_data():
    """Load and preprocess the heart disease dataset."""
    # Load the cleaned dataset
    df = pd.read_csv('../data/heart.csv')
    
    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def preprocess_data(df):
    """Preprocess the data and split into train/test sets."""
    # Define features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Define categorical and numerical features
    categorical_features = ['sex', 'chest_pain', 'fasting_blood_sugar', 'rest_ecg', 
                          'exercise_induced_angina', 'slope', 'num_major_vessels', 'thalassemia']
    numerical_features = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate', 'st_depression']
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, categorical_features, numerical_features

def create_preprocessor(categorical_features, numerical_features):
    """Create a preprocessing pipeline for the data."""
    # Create transformers for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_models(X_train, y_train, preprocessor):
    """Train multiple models and return the best one."""
    # Define models to train
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear', 'saga']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(
                random_state=42, 
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)
            ),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6, 9],
                'classifier__learning_rate': [0.01, 0.1, 0.3]
            }
        }
    }
    
    best_models = {}
    
    for model_name, model_info in models.items():
        print(f"\nTraining {model_name}...")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model_info['model'])
        ])
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_info['params'],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Store the best model
        best_models[model_name] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    return best_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models on the test set and return metrics."""
    results = []
    
    for model_name, model_info in models.items():
        model = model_info['model']
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, model_name)
        
        # Plot ROC curve
        plot_roc_curve(y_test, y_pred_proba, model_name)
        
        # Plot precision-recall curve
        plot_precision_recall_curve(y_test, y_pred_proba, model_name)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).set_index('Model')
    
    return results_df

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'../reports/figures/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'../reports/figures/roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, model_name):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', label=f'AP = {average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'../reports/figures/precision_recall_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def save_best_model(best_models, results_df):
    """Save the best model based on ROC AUC score."""
    # Get the best model name
    best_model_name = results_df['ROC AUC'].idxmax()
    best_model = best_models[best_model_name]['model']
    
    # Save the model
    model_path = '../app/model.pkl'
    joblib.dump(best_model, model_path)
    
    # Save feature names for later use in the app
    feature_names = list(X_train.columns)
    with open('../app/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    print(f"\nBest model saved: {best_model_name}")
    print(f"Model saved to: {os.path.abspath(model_path)}")
    
    return best_model_name

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    X_train, X_test, y_train, y_test, categorical_features, numerical_features = preprocess_data(df)
    
    # Create preprocessor
    preprocessor = create_preprocessor(categorical_features, numerical_features)
    
    # Train models
    print("\nTraining models...")
    best_models = train_models(X_train, y_train, preprocessor)
    
    # Evaluate models
    print("\nEvaluating models on test set...")
    results_df = evaluate_models(best_models, X_test, y_test)
    
    # Display results
    print("\nModel Performance on Test Set:")
    print(results_df)
    
    # Save the best model
    best_model_name = save_best_model(best_models, results_df)
    print(f"\nBest model based on ROC AUC: {best_model_name}")
    
    # Save results to CSV
    results_df.to_csv('../reports/model_results.csv')
    print("\nModel evaluation results saved to: ../reports/model_results.csv")

if __name__ == "__main__":
    main()
