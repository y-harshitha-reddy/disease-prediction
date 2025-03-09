import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, confusion_matrix

# Load data
data = pd.read_excel(r"disease_trends_india_cleaned_encoded.xlsx")

# Preprocess data for regression/classification
def preprocess_data(data, model_type):
    label_encoders = {}
    categorical_columns = ['State/Region', 'Disease Name', 'Season/Month', 'Age Group Affected', 
                           'Gender Distribution', 'Urban/Rural Classification', 'Comorbidities', 
                           'Lockdown Measures', 'Vaccination Drive']
    
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
    
    scaler = StandardScaler()
    numerical_columns = ['Year', 'Vaccination Coverage (%)', 'Monthly Cases', 'Temperature (°C)', 
                         'Humidity (%)', 'Rainfall (mm)', 'Air Pollution (PM2.5)', 
                         'Hospital Beds Available', 'Doctors per 10,000', 
                         'Population Density (per sq.km)', 'Number of Deaths']
    
    available_numerical = [col for col in numerical_columns if col in data.columns]
    if not available_numerical:
        st.error("No numerical columns available for scaling.")
        return None, None, None, None
    data[available_numerical] = scaler.fit_transform(data[available_numerical])
    
    X = data[available_numerical + [col for col in categorical_columns if col in data.columns]]

    # Fix target variable
    if model_type == "classification":
        y = data['Number of Cases'].apply(lambda x: "High" if x > data['Number of Cases'].median() else "Low")
    elif model_type == "regression":
        y = data['Number of Cases'].astype(float)
    else:
        st.error("Invalid model type. Choose 'classification' or 'regression'.")
        return None, None, None, None
    
    y.fillna(y.mode()[0] if model_type == "classification" else y.mean(), inplace=True)
    
    return X, y, scaler, label_encoders

# Train models
def train_model(X, y, model_choice, model_type):
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=42) if model_type == "classification" else DecisionTreeRegressor(max_depth=5, random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42) if model_type == "classification" else RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    elif model_choice == "Naïve Bayes" and model_type == "classification":
        model = GaussianNB()
    elif model_choice == "Linear Regression" and model_type == "regression":
        model = LinearRegression()
    else:
        st.error("Invalid model selection for the chosen type.")
        return None, None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

# Sidebar for model selection
st.sidebar.header("🔍 Disease Prediction")
model_choice = st.sidebar.radio("Select Model", ["Decision Tree", "Random Forest", "Naïve Bayes", "Linear Regression"])
model_type = st.sidebar.radio("Select Model Type", ["Classification", "Regression"])

features = ['Year', 'Vaccination Coverage (%)', 'Monthly Cases', 'Temperature (°C)', 
            'Humidity (%)', 'Rainfall (mm)', 'Air Pollution (PM2.5)', 
            'Hospital Beds Available', 'Doctors per 10,000', 
            'Population Density (per sq.km)', 'Number of Deaths']
selected_features = st.sidebar.multiselect("Select Features", features, default=features)

st.title("🏥 Predict the Rise of Diseases Based on Historical Health Records")

if selected_features:
    X, y, scaler, label_encoders = preprocess_data(data, model_type.lower())
    if X is None:
        st.error("Data preprocessing failed. Please check your dataset.")
    else:
        X = X[selected_features]
        model, X_test, y_test, y_pred = train_model(X, y, model_choice, model_type.lower())
        
        if model_type.lower() == "regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.subheader("📊 Model Performance")
            st.write(f"📉 MSE: {mse:.2f}")
            st.write(f"📈 R² Score: {r2:.2f}")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel("Actual Number of Cases")
            plt.ylabel("Predicted Number of Cases")
            plt.title("Predicted vs Actual Number of Cases")
            st.pyplot(plt)
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            st.subheader("📊 Model Performance")
            st.write(f"🎯 Accuracy: {accuracy:.2%}")
            st.write(f"📌 Precision: {precision:.2f}")
            st.write(f"📌 Recall: {recall:.2f}")
            cm = confusion_matrix(y_test, y_pred, labels=["Low", "High"])
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            st.pyplot(plt)
else:
    st.warning("Please select at least one feature to proceed.")
