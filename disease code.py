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
data = pd.read_excel(r"C:\Users\dhoni\Videos\AIML\disease_trends_india_cleaned_encoded.xlsx")

# Keep a copy of the original, unscaled data for slider ranges
original_data = data.copy()

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

# Sidebar for feature selection (used in Tab 1)
st.sidebar.header("🔍 Disease Rise Prediction (Tab 1)")
features = ['Year', 'Vaccination Coverage (%)', 'Monthly Cases', 'Temperature (°C)', 
            'Humidity (%)', 'Rainfall (mm)', 'Air Pollution (PM2.5)', 
            'Hospital Beds Available', 'Doctors per 10,000', 
            'Population Density (per sq.km)', 'Number of Deaths']
selected_features = st.sidebar.multiselect("Select Features for Prediction", features, default=features)

# Create tabs
tab1, tab2 = st.tabs(["📈 Predict Disease Rise", "📝 Input Values and Predict"])

# Tab 1: Predict the Rise of Diseases (Regression) - Unchanged
with tab1:
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
            historical_average = original_data['Number of Cases'].mean()  # Use original data
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

# Tab 2: Input Values and Predict - Corrected to depend only on user inputs
with tab2:
    st.title("🏥 Input Values to Predict Disease Cases")
    st.markdown("""
    In this tab, you can input specific values for the features to predict the number of disease cases using the trained model from the first tab. 
    The sliders below show the original values (e.g., Temperature in °C, not normalized values), making it easy for anyone to use. 
    The prediction is based solely on your input values, not the original dataset. 
    *Note:* Categorical features (e.g., State/Region) are shown as numbers because the dataset provided is already encoded. In a real application, these would be names like 'Maharashtra' or 'Kerala'.
    """)
    
    if selected_features:
        # Ensure the model and preprocessing objects are available from Tab 1
        if 'reg' not in locals() or 'scaler_reg' not in locals() or 'label_encoders_reg' not in locals():
            st.error("Please run the model in the 'Predict Disease Rise' tab first to train the model.")
        else:
            st.subheader("📝 Input Feature Values")
            
            # Create sliders for numerical features using original values
            numerical_inputs = {}
            numerical_columns = ['Year', 'Vaccination Coverage (%)', 'Monthly Cases', 'Temperature (°C)', 
                                 'Humidity (%)', 'Rainfall (mm)', 'Air Pollution (PM2.5)', 
                                 'Hospital Beds Available', 'Doctors per 10,000', 
                                 'Population Density (per sq.km)', 'Number of Deaths']
            
            for col in numerical_columns:
                if col in selected_features:
                    # Use original_data to get unscaled ranges
                    min_val = float(original_data[col].min())
                    max_val = float(original_data[col].max())
                    avg_val = float(original_data[col].mean())
                    # Adjust step size for better granularity
                    step_size = 1.0 if col == 'Year' else (max_val - min_val) / 100
                    numerical_inputs[col] = st.slider(f"{col}", min_val, max_val, avg_val, step=step_size)
            
            # Create dropdowns for categorical features
            categorical_inputs = {}
            categorical_columns = ['State/Region', 'Disease Name', 'Season/Month', 'Age Group Affected', 
                                   'Gender Distribution', 'Urban/Rural Classification', 'Comorbidities', 
                                   'Lockdown Measures', 'Vaccination Drive']
            
            for col in categorical_columns:
                if col in selected_features:
                    unique_vals = sorted(original_data[col].unique())
                    categorical_inputs[col] = st.selectbox(f"{col} (Encoded Values)", unique_vals)
            
            # Predict button
            if st.button("Predict Number of Cases"):
                # Prepare input data
                input_data = {}
                
                # Add numerical inputs (original values)
                for col in numerical_columns:
                    if col in selected_features:
                        input_data[col] = numerical_inputs[col]
                
                # Add categorical inputs (already encoded)
                for col in categorical_columns:
                    if col in selected_features:
                        input_data[col] = categorical_inputs[col]
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Debug: Display input data before scaling
                st.write("*Debug: Input Data Before Scaling*", input_df)
                
                # Scale numerical features using the same scaler from Tab 1
                numerical_cols_to_scale = [col for col in numerical_columns if col in selected_features]
                if numerical_cols_to_scale:
                    # Create a temporary DataFrame with all selected features
                    temp_df = pd.DataFrame(index=[0], columns=selected_features)
                    for col in selected_features:
                        if col in numerical_cols_to_scale:
                            temp_df[col] = input_df[col]
                        else:
                            temp_df[col] = input_df[col]  # Categorical values are already encoded
                    # Scale only numerical columns
                    temp_df[numerical_cols_to_scale] = scaler_reg.transform(temp_df[numerical_cols_to_scale])
                    input_df = temp_df
                
                # Debug: Display input data after scaling
                st.write("*Debug: Input Data After Scaling*", input_df)
                
                # Make prediction
                try:
                    predicted_cases = reg.predict(input_df)[0]
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.stop()
                
                # Display prediction (based only on input values)
                st.subheader("📈 Prediction Result")
                st.write(f"*Predicted Number of Cases:* {predicted_cases:.0f}")
                st.markdown("""
                #### Explanation:
                - *Predicted Number of Cases:* The model forecasts this number of cases based solely on the values you input.
                - *Healthcare Insight:* Use this prediction to assess the potential impact of the conditions you specified (e.g., high temperature or low vaccination coverage).
                """)
                
                # Insight Based on Prediction
                st.subheader("💡 Insight Based on Prediction")
                if predicted_cases > 15000:  # Arbitrary threshold based on dataset max
                    st.write("""
                    *Critical Alert:* A predicted case count above 15,000 suggests a significant outbreak. Immediate action is recommended, including:
                    - Increasing hospital bed capacity.
                    - Deploying additional medical staff and supplies.
                    - Launching public health campaigns.
                    """)
                elif 10000 <= predicted_cases <= 15000:
                    st.write("""
                    *Moderate Concern:* A predicted case count of 10,000-15,000 indicates a moderate outbreak. Consider:
                    - Monitoring healthcare facilities for capacity.
                    - Preparing contingency plans.
                    - Enhancing preventive measures.
                    """)
                elif 5000 <= predicted_cases < 10000:
                    st.write("""
                    *Stable with Moderate Cases:* A predicted case count of 5,000-10,000 suggests a manageable situation. Maintain current resources and monitor trends.
                    """)
                elif 1000 <= predicted_cases < 5000:
                    st.write("""
                    *Stable with Low Cases:* A predicted case count of 1,000-5,000 suggests cases are under control. Continue current interventions.
                    """)
                else:  # predicted_cases < 1000
                    st.write("""
                    *Positive Trend:* A predicted case count below 1,000 is a positive sign. Consider reallocating resources or scaling back interventions.
                    """)
    else:
        st.warning("Please select at least one feature from the sidebar to input values.")

# FAQ Section - Unchanged
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
