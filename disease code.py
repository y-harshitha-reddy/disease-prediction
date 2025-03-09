import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

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
        else:
            st.write(f"Warning: Column '{col}' not found in dataset. Skipping encoding.")
    
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
        y = data['Number of Cases'].apply(lambda x: "High" if x > data['Number of Cases'].median() else "Low")  # Convert to categorical
    elif model_type == "regression":
        y = data['Number of Cases'].astype(float)  # Convert to float for regression
    else:
        st.error("Invalid model type. Choose 'classification' or 'regression'.")
        return None, None, None, None
    
    # Handle missing values
    y.fillna(y.mode()[0] if model_type == "classification" else y.mean(), inplace=True)
    
    return X, y, scaler, label_encoders

# Train decision tree model
def train_decision_tree(X, y, model_type):
    if model_type == "classification":
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_type == "regression":
        clf = DecisionTreeRegressor(max_depth=5, random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'classification' or 'regression'.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf, X_test, y_test

# Sidebar for feature selection
st.sidebar.header("🔍 Disease Rise Prediction")
model_type = st.sidebar.radio("Select Model Type", ["Classification", "Regression"])

features = ['Year', 'Vaccination Coverage (%)', 'Monthly Cases', 'Temperature (°C)', 
            'Humidity (%)', 'Rainfall (mm)', 'Air Pollution (PM2.5)', 
            'Hospital Beds Available', 'Doctors per 10,000', 
            'Population Density (per sq.km)', 'Number of Deaths']
selected_features = st.sidebar.multiselect("Select Features for Prediction", features, default=features)

# Main content
st.title("🏥 Predict the Rise of Diseases Based on Historical Health Records")

if selected_features:
    X, y, scaler, label_encoders = preprocess_data(data, model_type.lower())
    if X is None:
        st.error("Data preprocessing failed. Please check your dataset.")
    else:
        X = X[selected_features]
        clf, X_test, y_test = train_decision_tree(X, y, model_type.lower())
        
        if model_type.lower() == "regression":
            y_pred = clf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.subheader("📊 Model Performance")
            st.write(f"📉 *Mean Squared Error (MSE):* {mse:.2f}")
            st.write(f"📈 *R² Score:* {r2:.2f}")
            
            # Predicted vs Actual Visualization
            st.subheader("📉 Predicted vs Actual Number of Cases")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel("Actual Number of Cases")
            plt.ylabel("Predicted Number of Cases")
            plt.title("Predicted vs Actual Number of Cases")
            st.pyplot(plt)

        else:  # Classification
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("📊 Model Performance")
            st.write(f"🎯 *Accuracy:* {accuracy:.2%}")

            # Confusion Matrix
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            st.pyplot(plt)

        # Feature Importance
        st.subheader("🔥 Feature Importance")
        feature_importance = pd.DataFrame({
            "Feature": selected_features,
            "Importance": clf.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
        plt.title("Feature Importance in Predicting Disease Cases")
        st.pyplot(plt)

else:
    st.warning("Please select at least one feature from the sidebar to run the model.")
