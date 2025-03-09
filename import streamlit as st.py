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
    df = pd.read_excel("endangered_species_dataset.xlsx")  # Ensure the file exists
    return df

df = load_data()

# Preprocessing function
def preprocess_data(df):
    label_encoders = {}
    categorical_columns = ["Region", "Habitat_Type"]

    # Encode categorical features
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_columns = [
        "Current_Population", "Population_Decline_Rate (%)",
        "Average_Temperature (°C)", "Climate_Change_Risk (%)",
        "Fragmentation_Risk (%)"
    ]
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Encode target variable (Extinction Risk)
    if "Extinction_Risk (%)" in df.columns:
        bins = [0, 33, 66, 100]
        labels = ["Low", "Medium", "High"]
        df["Extinction_Risk_Class"] = pd.cut(df["Extinction_Risk (%)"], bins=bins, labels=labels)
        le_risk = LabelEncoder()
        df["Extinction_Risk_Class"] = le_risk.fit_transform(df["Extinction_Risk_Class"])
    else:
        raise ValueError("Column 'Extinction_Risk (%)' not found in dataset!")

    X = df.drop(columns=["Species", "Extinction_Risk (%)", "Image_URL"], errors="ignore")
    y = df["Extinction_Risk_Class"]

    return X, y, scaler, label_encoders

# Sidebar selection
st.sidebar.title("🔬 Disease Prediction Dashboard")
option = st.sidebar.radio("Select Model Type", ["Classification", "Regression"])

# Display dataset
st.write("## 📊 Dataset Overview")
st.dataframe(df.head())

# Preprocess data
X, y_classification, scaler, label_encoders = preprocess_data(df)
y_regression = df["Current_Population"]  # Regression target

# Split dataset
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_classification, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Handle missing values
X_train_c.fillna(X_train_c.median(), inplace=True)
X_test_c.fillna(X_test_c.median(), inplace=True)

# Model selection
if option == "Classification":
    st.write("## 🤖 Extinction Risk Classification Model")

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
    st.write("## 📈 Population Prediction Model")

    reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)

    st.write("### R2 Score:", r2_score(y_test_r, y_pred_r))
    st.write("### Mean Squared Error:", mean_squared_error(y_test_r, y_pred_r))

    # Scatter Plot (Actual vs Predicted)
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.scatter(y_test_r, y_pred_r, alpha=0.5)
    plt.xlabel("Actual Population")
    plt.ylabel("Predicted Population")
    plt.title("Actual vs Predicted Population")
    st.pyplot(fig)

# Feature Importance
st.write("## 🔥 Feature Importance")
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(feature_importance)

# Predicting Future Trends
st.write("## 🔮 Future Prediction of Extinction Risk")
future_years = 5

for species in df["Species"].unique():
    species_df = df[df["Species"] == species]

    if species_df["Current_Population"].dtype == "O":  # Convert non-numeric cases
        species_df["Current_Population"] = pd.to_numeric(species_df["Current_Population"], errors="coerce")

    model = ExponentialSmoothing(species_df["Current_Population"], trend="add", seasonal=None, damped_trend=True)
    fitted_model = model.fit()
    future_predictions = fitted_model.forecast(future_years)

    st.write(f"### {species} - Predicted Population for the Next {future_years} Years")
    future_index = list(range(df["Year"].max() + 1, df["Year"].max() + 1 + future_years))
    pred_df = pd.DataFrame({"Year": future_index, "Predicted Population": future_predictions})
    st.dataframe(pred_df)

    # Line Plot for Predictions
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=df["Year"], y=df["Current_Population"], data=species_df, marker="o", label="Historical Data")
    sns.lineplot(x=future_index, y=future_predictions, marker="o", linestyle="dashed", label="Predicted Data")
    plt.title(f"{species} - Future Predictions")
    st.pyplot(fig)

st.sidebar.write("👨‍💻 Developed for Streamlit 🚀")
