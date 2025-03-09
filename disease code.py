import os
file_path = r"C:\Users\91965\Downloads\disease_trends_india_cleaned_encoded.xlsx"

if os.path.exists(file_path):
    print("File found!")
else:
    print("File not found. Check the path again.")
