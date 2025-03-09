import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

def preprocess_data(data):
    label_encoders = {}
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = data[col].astype(str).fillna("Unknown")  # Handling NaN values before encoding
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    scaler = StandardScaler()
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns].fillna(data[numerical_columns].median()))
    
    X = data.drop(columns=['Extinction_Risk (%)'], errors='ignore')  # Assuming this is the target
    y = data['Extinction_Risk (%)'] if 'Extinction_Risk (%)' in data.columns else None
    
    return X, y, scaler, label_encoders, numerical_columns, categorical_columns

def compute_information_gain(model, X, y):
    return dict(zip(X.columns, model.feature_importances_))

# Load and preprocess data
data = load_data("/mnt/data/disease_trends_india_cleaned_encoded.xlsx")
X, y, scaler, label_encoders, numerical_columns, categorical_columns = preprocess_data(data)

st.sidebar.header("🔍 Decision Tree Model")
features = X.columns.tolist()
selected_features = st.sidebar.multiselect("Select Features for Decision Tree", features, default=features)

if selected_features:
    X = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    IG_results = compute_information_gain(clf, X_train, y_train)

    st.sidebar.subheader("📊 Information Gain for Each Feature")
    for feature, ig in IG_results.items():
        st.sidebar.write(f"{feature}: **{ig:.4f}**")

    st.subheader("🔍 Insights from Information Gain Analysis (Decision Tree)")
    if IG_results:
        best_feature = max(IG_results, key=IG_results.get)
        worst_feature = min(IG_results, key=IG_results.get)
        st.write(f"✅ The *most important feature* is *{best_feature}* with an IG of *{IG_results[best_feature]:.4f}*.")
        st.write(f"⚠ The *least important feature* is *{worst_feature}* with an IG of *{IG_results[worst_feature]:.4f}*.")
        st.write("🔹 Features with *higher IG values* contribute more to the decision-making process.")
    else:
        st.write("⚠ No valid Information Gain could be calculated. Check the dataset!")
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    st.sidebar.subheader("📊 Decision Tree Performance")
    st.sidebar.write(f"*Accuracy:* {accuracy:.2f}")
    st.sidebar.write(f"*Precision:* {precision:.2f}")
    st.sidebar.write(f"*Recall:* {recall:.2f}")
    
    st.subheader("📌 Decision Tree Visualization")
    plt.figure(figsize=(12, 6))
    plot_tree(clf, feature_names=selected_features, filled=True)
    st.pyplot(plt)
    
    st.subheader("💡 Feature Importance")
    feature_importance = pd.DataFrame({
        "Feature": selected_features,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
    plt.title("Feature Importance in Decision Tree")
    st.pyplot(plt)
