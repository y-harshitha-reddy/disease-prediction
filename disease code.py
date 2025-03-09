import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_excel(r"disease_trends_india_cleaned_encoded.xlsx")

# Preprocess data for regression
def preprocess_data_regression(data):
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
    y = data['Number of Cases']  # Target for regression
    
    return X, y, scaler, label_encoders

# Sidebar for feature selection
st.sidebar.header("🔍 Disease Rise Prediction")
features = ['Year', 'Vaccination Coverage (%)', 'Monthly Cases', 'Temperature (°C)', 
            'Humidity (%)', 'Rainfall (mm)', 'Air Pollution (PM2.5)', 
            'Hospital Beds Available', 'Doctors per 10,000', 
            'Population Density (per sq.km)', 'Number of Deaths']
selected_features = st.sidebar.multiselect("Select Features for Prediction", features, default=features)

# Main content
st.title("🏥 Predict the Rise of Diseases Based on Historical Health Records")
st.markdown("""
This application uses a *Decision Tree Regression model* to predict the number of disease cases based on historical health records from India. 
The goal is to forecast potential rises in disease cases, enabling healthcare professionals to plan resources effectively. 
Select features in the sidebar to customize the model and explore their impact.
""")

if selected_features:
    X_reg, y_reg, scaler_reg, label_encoders_reg = preprocess_data_regression(data)
    if X_reg is None:
        st.error("Data preprocessing failed. Please check your dataset.")
    else:
        X_reg = X_reg[selected_features]
        
        # Split and train the model
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        reg = DecisionTreeRegressor(max_depth=5, random_state=42)
        reg.fit(X_train_reg, y_train_reg)
        
        y_pred_reg = reg.predict(X_test_reg)
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)
        
        # Calculate rise indicator
        historical_average = data['Number of Cases'].mean()
        predicted_average = y_pred_reg.mean()
        rise_percentage = ((predicted_average - historical_average) / historical_average) * 100 if historical_average > 0 else 0
        
        # Model Performance
        st.subheader("📊 Model Performance")
        st.write(f"📉 *Mean Squared Error (MSE):* {mse:.2f}")
        st.write(f"📈 *R² Score:* {r2:.2f}")
        st.markdown("""
        #### Explanation:
        - *Mean Squared Error (MSE):* Measures the average error in predicting the number of cases. A lower value (e.g., <1000) means predictions are closer to actual values, critical for accurate healthcare planning.
        - *R² Score:* Shows how well the model explains variations in case numbers. A value near 1 (e.g., 0.8) indicates the model captures key factors driving disease rise.
        - *Healthcare Insight:* Accurate predictions (low MSE, high R²) help hospitals prepare for surges in cases, ensuring sufficient staff and supplies.
        """)
        
        # Predicted vs Actual Visualization
        st.subheader("📉 Predicted vs Actual Number of Cases")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_reg, y_pred_reg, color='blue', alpha=0.5)
        plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
        plt.xlabel("Actual Number of Cases")
        plt.ylabel("Predicted Number of Cases")
        plt.title("Predicted vs Actual Number of Cases")
        st.pyplot(plt)
        st.markdown("""
        #### Explanation:
        - *Scatter Plot:* Each point compares actual cases (x-axis) to predicted cases (y-axis).
        - *Red Line:* Represents perfect predictions (predicted = actual). Points near this line are accurate.
        - *Healthcare Insight:* If points are clustered near the line, the model reliably forecasts case numbers, aiding in timely interventions like vaccination drives or hospital bed allocation.
        """)
        
        # Feature Importance
        st.subheader("🔥 Feature Importance")
        feature_importance_reg = pd.DataFrame({
            "Feature": selected_features,
            "Importance": reg.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_reg, palette="viridis")
        plt.title("Feature Importance in Predicting Disease Cases")
        st.pyplot(plt)
        st.markdown("""
        #### Explanation:
        - *Bar Chart:* Shows which features most influence predictions. Higher bars indicate greater impact.
        - *Example:* If Vaccination Coverage (%) has high importance, it suggests vaccination rates strongly affect case numbers—a key insight for public health.
        - *Healthcare Insight:* Focus on high-importance features (e.g., Air Pollution (PM2.5)) for targeted interventions, like improving air quality to reduce disease spread.
        """)
        
        # Rise Indicator
        st.subheader("📈 Predicted Rise in Disease Cases")
        st.write(f"*Historical Average Cases:* {historical_average:.0f}")
        st.write(f"*Predicted Average Cases:* {predicted_average:.0f}")
        st.write(f"*Predicted Rise:* {rise_percentage:.2f}%")
        st.markdown("""
        #### Explanation:
        - *Historical Average:* The baseline number of cases from past data.
        - *Predicted Average:* The model’s forecast for future cases based on test data.
        - *Rise Percentage:* A positive value (e.g., 15%) indicates a predicted increase, while a negative value suggests a decline.
        - *Healthcare Insight:* A 20% rise might signal the need for extra hospital beds or staff, helping administrators act proactively to save lives.
        """)

else:
    st.warning("Please select at least one feature from the sidebar to run the model.")

# FAQ Section
st.subheader("❓ Frequently Asked Questions")
faq_data = {
    "What does this app do?": "It predicts the rise of disease cases using historical health records, helping healthcare professionals plan resources.",
    "How does it predict the rise?": "A Decision Tree Regressor analyzes selected features (e.g., temperature, vaccination rates) to forecast case numbers.",
    "Why is feature importance shown?": "It identifies which factors (e.g., pollution, hospital beds) most affect disease rise, guiding targeted interventions.",
    "What does a predicted rise mean?": "A percentage increase (e.g., 10%) suggests more cases than the historical average, signaling a potential outbreak.",
    "Can I trust the output?": "The model provides estimates based on past data. Use it as a planning tool alongside expert judgment, not as a definitive prediction."
}
for question, answer in faq_data.items():
    with st.expander(f"▶ {question}"):
        st.markdown(f"*Answer:* {answer}")
