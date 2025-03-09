import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    data = pd.read_excel(file_path, sheet_name="Sheet1")
    return data

def preprocess_data(data):
    label_encoders = {}
    categorical_columns = ['State/Region', 'Season/Month', 'Age Group Affected', 'Gender Distribution', 
                           'Urban/Rural Classification', 'Lockdown Measures', 'Vaccination Drive']
    
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))  # Convert to string before encoding
        label_encoders[col] = le
    
    scaler = StandardScaler()
    numerical_columns = ['Year', 'Number of Cases', 'Number of Deaths', 'Vaccination Coverage (%)', 'Monthly Cases',
                         'Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)', 'Air Pollution (PM2.5)',
                         'Hospital Beds Available', 'Doctors per 10,000', 'Population Density (per sq.km)']
    
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    X = data.drop(columns=['Disease Name'], errors='ignore')  # Target variable not defined, assuming Disease Name is not a feature
    
    return X, scaler, label_encoders, numerical_columns, categorical_columns

# Load and preprocess
data = load_data("/mnt/data/disease_trends_india_cleaned.xlsx")
X, scaler, label_encoders, numerical_columns, categorical_columns = preprocess_data(data)
