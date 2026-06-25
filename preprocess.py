import pandas as pd
import numpy as np
import os

def clean_dataset():
    raw_path = "./data/raw"
    processed_path = "./data/processed"
    os.makedirs(processed_path, exist_ok=True)
    
    print("--- Loading Raw Datasets ---")
    train = pd.read_csv(f"{raw_path}/train.csv")
    test = pd.read_csv(f"{raw_path}/test.csv")
    
    print(f"Original shapes -> Train: {train.shape}, Test: {test.shape}")
    
    # 1. Drop features with zero variance or identifiers
    # contry_of_res is 100% 'Unknown' -> drop
    # used_app_before is 100% 'no' -> drop
    # age_desc is constant in train/test -> drop
    # ID is row identifier -> drop
    # result is perfect collinearity with A1-A10 sum -> drop
    cols_to_drop = ['ID', 'age_desc', 'contry_of_res', 'used_app_before', 'result']
    print(f"Dropping columns: {cols_to_drop}")
    train = train.drop(columns=[c for c in cols_to_drop if c in train.columns])
    test = test.drop(columns=[c for c in cols_to_drop if c in test.columns])
    
    # 2. Clean binary features (map 'yes' to 1, 'no' to 0)
    print("Encoding binary features (jaundice, austim, gender)...")
    binary_cols = ['jaundice', 'austim']
    for col in binary_cols:
        train[col] = train[col].map({'yes': 1, 'no': 0})
        test[col] = test[col].map({'yes': 1, 'no': 0})
        
    train['gender'] = train['gender'].apply(lambda x: 1 if str(x).strip().lower() == 'm' else 0)
    test['gender'] = test['gender'].apply(lambda x: 1 if str(x).strip().lower() == 'm' else 0)
    
    # 3. Clean ethnicity typos
    print("Standardizing 'ethnicity' categories...")
    def clean_ethnicity(val):
        if pd.isna(val) or val == '?':
            return 'Others'
        val = val.strip().replace('-', ' ').lower()
        if val == 'others':
            return 'Others'
        return val.title()
        
    train['ethnicity'] = train['ethnicity'].apply(clean_ethnicity)
    test['ethnicity'] = test['ethnicity'].apply(clean_ethnicity)
    
    # 4. Handle Age outliers (e.g. 383.0 years)
    print("Handling age outliers...")
    train['age'] = pd.to_numeric(train['age'], errors='coerce')
    test['age'] = pd.to_numeric(test['age'], errors='coerce')
    
    # Calculate median of train age
    median_age = train['age'].median()
    
    # Cap age > 100 to median
    train.loc[train['age'] > 100, 'age'] = median_age
    test.loc[test['age'] > 100, 'age'] = median_age
    
    # Fill remaining NaNs if any
    train['age'] = train['age'].fillna(median_age)
    test['age'] = test['age'].fillna(median_age)
    
    # 5. One-Hot Encoding for categorical features (relation, ethnicity)
    # One-hot encoding is 100% data leakage-free as it doesn't depend on the target
    print("One-hot encoding categorical features (ethnicity, relation)...")
    categorical_cols = ['ethnicity', 'relation']
    
    # Remember target column name
    target_col = 'Class/ASD'
    target_series = train[target_col].copy()
    
    # Temporarily drop target for one-hot encoding alignment
    X_train = train.drop(columns=[target_col])
    X_test = test.copy()
    
    # Apply get_dummies
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Align train and test columns (fill missing columns in test with 0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Re-attach target
    train_cleaned = X_train.copy()
    train_cleaned[target_col] = target_series
    test_cleaned = X_test.copy()
    
    print(f"Final cleaned shapes -> Train: {train_cleaned.shape}, Test: {test_cleaned.shape}")
    
    # Save to processed folder
    train_cleaned.to_csv(f"{processed_path}/train_cleaned.csv", index=False)
    test_cleaned.to_csv(f"{processed_path}/test_cleaned.csv", index=False)
    print("Cleaned datasets saved successfully!")

if __name__ == "__main__":
    clean_dataset()
