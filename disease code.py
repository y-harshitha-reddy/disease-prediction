import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, r2_score

def load_data(file_path):
    return pd.read_excel(file_path)

def preprocess_data(data):
    label_encoders = {}
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
    
    for col in categorical_columns:
        data[col] = data[col].fillna("Unknown")
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    return data, scaler, label_encoders

def train_decision_tree(X, y, model_type="classification"):
    if model_type == "classification":
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        clf = DecisionTreeRegressor(max_depth=5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    if model_type == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        return clf, accuracy, precision, recall
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return clf, mse, r2

def compute_information_gain(clf, X):
    return dict(zip(X.columns, clf.feature_importances_))

def main():
    st.title("📊 Disease & Species Risk Prediction Platform")
    file_path = "disease_trends_india_cleaned_encoded.xlsx"
    data = load_data(file_path)
    data, scaler, label_encoders = preprocess_data(data)
    
    model_choice = st.sidebar.radio("Select Model:", ["Decision Tree", "Random Forest", "Naïve Bayes", "Linear Regression"])
    
    if model_choice == "Decision Tree":
        st.sidebar.subheader("🔍 Decision Tree Settings")
        target = st.sidebar.selectbox("Select Target Variable", data.columns)
        features = st.sidebar.multiselect("Select Features", [col for col in data.columns if col != target])
        model_type = st.sidebar.radio("Model Type", ["Classification", "Regression"])
        
        if features:
            X, y = data[features], data[target]
            clf, metric1, metric2 = train_decision_tree(X, y, model_type.lower())
            st.write(f"Model Performance: {metric1:.2f}, {metric2:.2f}")
            
            st.subheader("📌 Decision Tree Visualization")
            plt.figure(figsize=(12, 6))
            plot_tree(clf, feature_names=features, filled=True)
            st.pyplot(plt)
            
            st.subheader("💡 Feature Importance")
            feature_importance = compute_information_gain(clf, X)
            feature_df = pd.DataFrame(feature_importance.items(), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
            sns.barplot(x="Importance", y="Feature", data=feature_df, palette="viridis")
            st.pyplot(plt)
    
    elif model_choice == "Random Forest":
        st.sidebar.subheader("🌲 Random Forest Settings")
        target = st.sidebar.selectbox("Select Target Variable", data.columns)
        features = st.sidebar.multiselect("Select Features", [col for col in data.columns if col != target])
        
        if features:
            X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
            rf_clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
            rf_clf.fit(X_train, y_train)
            y_pred_rf = rf_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_rf)
            st.write(f"Random Forest Accuracy: {accuracy:.2f}")
    
    elif model_choice == "Naïve Bayes":
        st.sidebar.subheader("🤖 Naïve Bayes Settings")
        target = st.sidebar.selectbox("Select Target Variable", data.columns)
        features = st.sidebar.multiselect("Select Features", [col for col in data.columns if col != target])
        
        if features:
            X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
            nb_clf = GaussianNB()
            nb_clf.fit(X_train, y_train)
            y_pred_nb = nb_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_nb)
            st.write(f"Naïve Bayes Accuracy: {accuracy:.2f}")
    
    elif model_choice == "Linear Regression":
        st.sidebar.subheader("📈 Linear Regression Settings")
        target = st.sidebar.selectbox("Select Target Variable", data.columns)
        features = st.sidebar.multiselect("Select Features", [col for col in data.columns if col != target])
        
        if features:
            X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Linear Regression MSE: {mse:.2f}, R² Score: {r2:.2f}")
    
if __name__ == "__main__":
    main()
