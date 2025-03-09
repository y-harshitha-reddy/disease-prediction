import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("C:\Users\91965\Desktop\disease prediction\disease_trends_india_updated.xlsx")
    return df

df = load_data()

# Sidebar
st.sidebar.title("🔬 Disease Prediction Tool")
option = st.sidebar.radio("Select Model Type", ["Classification", "Regression"])

# Show dataset
st.write("## 📊 Dataset Preview")
st.dataframe(df.head())

# Feature selection
features = ['Year', 'Number of Cases', 'Number of Deaths', 'Comorbidities', 'Monthly Cases']
X = df[features]

disease_mapping = {disease: idx for idx, disease in enumerate(df["Disease Name"].unique())}
df["Disease Label"] = df["Disease Name"].map(disease_mapping)

y_classification = df["Disease Label"]
y_regression = df["Number of Cases"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_classification, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_regression, test_size=0.2, random_state=42)

if option == "Classification":
    st.write("## 🤖 Decision Tree Classification Model")
    
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    
    st.write("### Accuracy Score:", accuracy_score(y_test_c, y_pred_c))
    st.text("Classification Report:")
    st.text(classification_report(y_test_c, y_pred_c))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, feature_names=features, class_names=list(disease_mapping.keys()), filled=True, fontsize=6)
    st.pyplot(fig)

elif option == "Regression":
    st.write("## 📈 Decision Tree Regression Model")
    
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

# Feature Importance
st.write("## 🔥 Feature Importance")
feature_importance = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
st.bar_chart(feature_importance)

# Correlation Heatmap
st.write("## 📌 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[features + ["Number of Cases"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(fig)

# Additional Visualizations
st.write("## 📊 Additional Data Visualizations")

# Line Plot for Yearly Disease Trends
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x="Year", y="Number of Cases", hue="Disease Name", marker="o")
plt.title("Yearly Disease Trends")
st.pyplot(fig)

# Boxplot for Monthly Cases Distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Disease Name", y="Monthly Cases")
plt.xticks(rotation=45)
plt.title("Monthly Cases Distribution by Disease")
st.pyplot(fig)

st.sidebar.write("👨‍💻 Developed for VS Code + Streamlit 🚀")
