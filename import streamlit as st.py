import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import entropy
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import LabelEncoder

# Ensure required packages are installed
try:
    import statsmodels
except ImportError:
    import os
    os.system("pip install statsmodels")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("disease_trends_india_updated.xlsx")
    return df

df = load_data()

# Encode categorical target variable
label_encoder = LabelEncoder()
y_classification = label_encoder.fit_transform(df["Disease Name"].astype(str))

df.fillna(method='ffill', inplace=True)  # Handle missing values

# Sidebar
st.sidebar.title("🔬 Disease Prediction Dashboard")
option = st.sidebar.radio("Select Model Type", ["Classification", "Regression"])

# Show dataset
st.write("## 📊 Dataset Overview")
st.dataframe(df.head())

# Feature selection
features = ['Year', 'Number of Cases', 'Number of Deaths', 'Monthly Cases']
X = df[features].fillna(df[features].median(numeric_only=True))
y_regression = df["Number of Cases"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_classification, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_regression, test_size=0.2, random_state=42)

if option == "Classification":
    st.write("## 🤖 Disease Classification Model")
    
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    
    st.write("### Accuracy Score:", accuracy_score(y_test_c, y_pred_c))
    st.text("Classification Report:")
    st.text(classification_report(y_test_c, y_pred_c))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, feature_names=features, filled=True, fontsize=6)
    st.pyplot(fig)

elif option == "Regression":
    st.write("## 📈 Disease Trend Prediction Model")
    
    reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)
    
    st.write("### R2 Score:", r2_score(y_test_r, y_pred_r))
    st.write("### Mean Squared Error:", mean_squared_error(y_test_r, y_pred_r))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.scatter(y_test_r, y_pred_r, alpha=0.5)
    plt.xlabel("Actual Cases")
    plt.ylabel("Predicted Cases")
    plt.title("Actual vs Predicted Cases")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    residuals = y_test_r - y_pred_r
    plt.scatter(y_pred_r, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='dashed')
    plt.xlabel("Predicted Cases")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    st.pyplot(fig)

# Feature Importance
st.write("## 🔥 Feature Importance")
feature_importance = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
st.bar_chart(feature_importance)

# Predicting Future Rise of Diseases
st.write("## 🔮 Future Prediction of Disease Trends")
future_years = 5
for disease in df["Disease Name"].unique():
    disease_df = df[df["Disease Name"] == disease]
    
    if disease_df["Number of Cases"].dtype == 'O':  # Convert non-numeric cases
        disease_df["Total Cases"] = pd.to_numeric(disease_df["Total Cases"], errors='coerce')
    
    model = ExponentialSmoothing(disease_df["Total Cases"], trend="add", seasonal=None, damped_trend=True)
    fitted_model = model.fit()
    future_predictions = fitted_model.forecast(future_years)
    
    st.write(f"### {disease} - Predicted Cases for the Next {future_years} Years")
    future_index = list(range(df["Year"].max() + 1, df["Year"].max() + 1 + future_years))
    pred_df = pd.DataFrame({"Year": future_index, "Predicted Cases": future_predictions})
    st.dataframe(pred_df)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=df["Year"], y=df["Total Cases"], data=disease_df, marker="o", label="Historical Data")
    sns.lineplot(x=future_index, y=future_predictions, marker="o", linestyle="dashed", label="Predicted Data")
    plt.title(f"{disease} - Future Predictions")
    st.pyplot(fig)

st.sidebar.write("👨‍💻 Developed for VS Code + Streamlit 🚀")
