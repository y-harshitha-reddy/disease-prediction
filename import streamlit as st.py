import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, mean_squared_error, r2_score,
    precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_excel("disease_trends_india_cleaned_encoded.xlsx")  # Ensure the file exists
    return df
    from sklearn.preprocessing import LabelEncoder

# List of categorical columns
categorical_columns = [
    "Disease Name", "State/Region", "Season/Month", "Age Group Affected",
    "Gender Distribution", "Urban/Rural Classification", "Comorbidities",
    "Lockdown Measures", "Vaccination Drive"
]

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string & encode
    label_encoders[col] = le

# Drop unnecessary columns
df.drop(columns=["Image URL"], inplace=True, errors="ignore")

# Now df is fully numeric and ready for model training


df = load_data()

# Preprocessing function
def preprocess_data(df):
    label_encoders = {}
    
    # List of categorical columns
    categorical_columns = [
        "Disease Name", "State/Region", "Season/Month", "Age Group Affected",
        "Gender Distribution", "Urban/Rural Classification", "Comorbidities",
        "Lockdown Measures", "Vaccination Drive"
    ]
    
    # Encode categorical features
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)  # Convert to string to prevent errors
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_columns = [
        "Year", "Number of Cases", "Number of Deaths", "Vaccination Coverage (%)",
        "Monthly Cases", "Temperature (°C)", "Humidity (%)", "Rainfall (mm)",
        "Air Pollution (PM2.5)", "Hospital Beds Available", "Doctors per 10,000",
        "Population Density (per sq.km)"
    ]
    
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Define risk bins and labels for disease rise classification
    if "Number of Cases" in df.columns:
        bins = [0, 1000, 10000, df["Number of Cases"].max()]
        labels = ["Low", "Medium", "High"]
        
        df["Disease_Rise_Risk"] = pd.cut(df["Number of Cases"], bins=bins, labels=labels)
        
        # Encode risk levels into numerical values
        le_risk = LabelEncoder()
        df["Disease_Rise_Risk"] = le_risk.fit_transform(df["Disease_Rise_Risk"])

    else:
        raise ValueError("Column 'Number of Cases' not found in dataset!")

    # Define features and target variable
    X = df.drop(columns=["Disease Name", "Number of Cases", "Image URL"], errors="ignore")
    y = df["Disease_Rise_Risk"]  # Target variable for classification

    return X, y, scaler, label_encoders

# Sidebar selection
st.sidebar.title("🔬 Disease Prediction Dashboard")
option = st.sidebar.radio("Select Model Type", ["Classification", "Regression"])

# Display dataset
st.write("## 📊 Dataset Overview")
st.dataframe(df.head())

# Preprocess data
X, y_classification, scaler, label_encoders = preprocess_data(df)
y_regression = df["Number of Cases"]

# Split dataset
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_classification, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Handle missing values
X_train_c.fillna(X_train_c.median(numeric_only=True), inplace=True)
X_test_c.fillna(X_test_c.median(numeric_only=True), inplace=True)

# Model selection
if option == "Classification":
    st.write("## 🤖 Disease Rise Classification Model")

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)

    st.write("### Accuracy Score:", accuracy_score(y_test_c, y_pred_c))
    st.text("Classification Report:")
    st.text(classification_report(y_test_c, y_pred_c))

    # Decision Tree Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, feature_names=X.columns, filled=True, fontsize=6)
    st.pyplot(fig)

elif option == "Regression":
    st.write("## 📈 Cases Prediction Model")

    reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)

    st.write("### R2 Score:", r2_score(y_test_r, y_pred_r))
    st.write("### Mean Squared Error:", mean_squared_error(y_test_r, y_pred_r))

    # Scatter Plot (Actual vs Predicted)
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.scatter(y_test_r, y_pred_r, alpha=0.5, color="blue")
    plt.xlabel("Actual Number of Cases")
    plt.ylabel("Predicted Number of Cases")
    plt.title("Actual vs Predicted Disease Cases")
    plt.grid(True)
    st.pyplot(fig)

# Feature Importance
st.write("## 🔥 Feature Importance")
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(feature_importance)

# Predicting Future Trends
st.write("## 🔮 Future Prediction of Disease Rise")
future_years = 5

for disease in df["Disease Name"].unique():
    disease_df = df[df["Disease Name"] == disease]

    # Ensure "Number of Cases" is numeric
    disease_df["Number of Cases"] = pd.to_numeric(disease_df["Number of Cases"], errors="coerce")

    # Apply Exponential Smoothing model for disease case forecasting
    model = ExponentialSmoothing(disease_df["Number of Cases"], trend="add", seasonal=None, damped_trend=True)
    fitted_model = model.fit()
    future_predictions = fitted_model.forecast(future_years)

    # Display Predictions
    st.write(f"### {disease} - Predicted Cases for the Next {future_years} Years")
    future_index = list(range(df["Year"].max() + 1, df["Year"].max() + 1 + future_years))
    pred_df = pd.DataFrame({"Year": future_index, "Predicted Cases": future_predictions})
    st.dataframe(pred_df)

    # Line Plot for Predictions
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=disease_df["Year"], y=disease_df["Number of Cases"], marker="o", label="Historical Data")
    sns.lineplot(x=future_index, y=future_predictions, marker="o", linestyle="dashed", label="Predicted Data")

    plt.title(f"{disease} - Future Case Predictions")
    plt.xlabel("Year")
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

st.sidebar.write("👨‍💻 Developed for Streamlit 🚀")
