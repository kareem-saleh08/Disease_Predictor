# ❤️ Heart Disease Prediction using Explainable AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-orange)

This project implements an end-to-end machine learning system for predicting heart disease with explainable AI techniques. The system uses the UCI Heart Disease dataset and provides interpretable predictions using SHAP and LIME, making the model's decisions transparent and understandable.

## 🚀 Features

- **Comprehensive Data Analysis**: In-depth exploratory data analysis with visualizations
- **Multiple ML Models**: Comparison of Logistic Regression, Random Forest, and XGBoost
- **Model Interpretability**: SHAP and LIME for explaining model predictions
- **Interactive Dashboard**: User-friendly Streamlit interface for predictions
- **Feature Importance**: Visual explanations of which factors contribute most to predictions
- **Responsive Design**: Works on both desktop and mobile devices

## 📋 Dataset

The dataset used in this project is the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), which contains 76 attributes, including the predicted attribute. The dataset contains 303 instances with no missing values after preprocessing.

### Features

1. **Demographic Information**:
   - Age: Age in years
   - Sex: Gender (1 = male; 0 = female)

2. **Medical History**:
   - Chest pain type (4 values)
   - Resting blood pressure (in mm Hg)
   - Serum cholesterol in mg/dl
   - Fasting blood sugar > 120 mg/dl
   - Resting electrocardiographic results (values 0,1,2)

3. **Exercise Test Results**:
   - Maximum heart rate achieved
   - Exercise induced angina (1 = yes; 0 = no)
   - ST depression induced by exercise relative to rest
   - The slope of the peak exercise ST segment

4. **Other Factors**:
   - Number of major vessels (0-3) colored by fluoroscopy
   - Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/early-disease-detection.git
   cd early-disease-detection
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Running the Streamlit App

To launch the interactive web application:

```bash
streamlit run app/app.py
```

Then open your browser and navigate to `http://localhost:8501`

### 2. Running Jupyter Notebooks

If you want to explore the data analysis, modeling, or explainability notebooks:

```bash
jupyter notebook notebooks/
```

## 📂 Project Structure

```
early-disease-detection/
│
├── data/                   # Dataset files
│   └── heart.csv           # Processed heart disease dataset
│
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_modeling.ipynb   # Model training and evaluation
│   └── 03_explainability.ipynb  # Model interpretability with SHAP and LIME
│
├── app/                    # Streamlit application
│   ├── app.py              # Main application file
│   ├── model.pkl           # Trained model
│   └── feature_names.json  # Feature names for the model
│
├── reports/                # Generated reports and visualizations
│   └── figures/            # Saved plots and charts
│
├── .gitignore              # Git ignore file
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## 📊 Results

The best performing model achieved the following metrics on the test set:

| Metric     | Score  |
|------------|--------|
| Accuracy   | 0.85   |
| Precision  | 0.88   |
| Recall     | 0.84   |
| F1-Score   | 0.86   |
| ROC-AUC    | 0.92   |

### Feature Importance

The most important features for predicting heart disease are:
1. Number of major vessels colored by fluoroscopy
2. Thalassemia
3. ST depression induced by exercise
4. Maximum heart rate achieved
5. Age

## 🤖 Model Interpretability

This project emphasizes model interpretability using SHAP and LIME:

- **SHAP (SHapley Additive exPlanations)**: Shows the contribution of each feature to the prediction
- **LIME (Local Interpretable Model-agnostic Explanations)**: Explains individual predictions


## 🙏 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for the Heart Disease dataset
- The SHAP and LIME communities for their excellent explainability tools
- Streamlit for making it easy to build interactive web apps
├── notebooks/          # Jupyter notebooks for EDA, modeling, and explainability
├── app/                # Streamlit application
├── reports/            # Generated reports and visualizations
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 📊 Dataset
This project uses the UCI Heart Disease dataset, which contains 76 attributes, including the predicted attribute. All existing patients are male, aged 29-77.
