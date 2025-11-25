import pandas as pd
import numpy as np
import os
import random

# --- CONFIGURATION ---
INPUT_FILE = 'raw_sales_data.csv'
OUTPUT_FILE = 'augmented_sales_data.csv'
TARGET_SIZE = 1000000  # Set your desired final number of rows here
NOISE_LEVEL = 0.05   # 5% variation added to numerical columns (Age, Price, etc.)

# Columns to treat as numerical for noise addition
NUMERICAL_COLS = ['Age', 'UnitPrice', 'Quantity', 'TotalPrice', 'ShippingFee']
# Column to ensure has unique, sequential identifiers
ID_COL = 'CustomerID'

def create_dummy_csv(filename):
    """Creates a small dummy CSV file if the input file doesn't exist."""
    print(f"Creating a placeholder file: {filename}")
    data = {
        'CustomerID': [f'CUST{i:04d}' for i in range(1, 11)],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South'],
        'Age': [25, 34, 51, 19, 42, 29, 60, 31, 22, 45],
        'ProductName': ['Monitor', 'Headphones', 'Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Smartwatch', 'Headphones', 'Laptop', 'Mouse'],
        'Category': ['Electronics', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Electronics', 'Wearables', 'Accessories', 'Electronics', 'Accessories'],
        'UnitPrice': [300.00, 15.00, 1200.00, 10.00, 50.00, 320.00, 250.00, 18.00, 1100.00, 9.50],
        'Quantity': [1, 2, 1, 3, 2, 1, 1, 4, 1, 3],
        'TotalPrice': [300.00, 30.00, 1200.00, 30.00, 100.00, 320.00, 250.00, 72.00, 1100.00, 28.50],
        'ShippingFee': [13.31, 6.93, 11.31, 10.70, 5.00, 13.50, 11.20, 7.50, 10.50, 6.80],
        'ShippingStatus': ['Returned', 'In Transit', 'Delivered', 'Delivered', 'Returned', 'Delivered', 'In Transit', 'Returned', 'Delivered', 'In Transit'],
        'OrderDate': ['2023-12-08', '2023-04-09', '2023-08-28', '2023-01-18', '2023-01-19', '2023-02-01', '2023-03-10', '2023-04-22', '2023-05-15', '2023-06-01']
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def generate_unique_ids(df_augmented, start_id_num):
    """Generates new sequential CustomerIDs for the augmented data."""
    num_rows = len(df_augmented)
    new_ids = []
    
    # Assuming the format is 'CUST' followed by a zero-padded number (e.g., CUST0001)
    for i in range(num_rows):
        new_id = f'CUST{start_id_num + i:04d}'
        new_ids.append(new_id)

    df_augmented[ID_COL] = new_ids
    return df_augmented


def add_random_noise(df_augmented, cols, noise_level):
    """Adds a small, normally distributed noise to numerical columns."""
    for col in cols:
        if col in df_augmented.columns:
            # Calculate the standard deviation for the noise, based on the mean value and noise level
            std_dev = df_augmented[col].mean() * noise_level 
            
            # Generate random noise from a normal distribution
            noise = np.random.normal(loc=0, scale=std_dev, size=len(df_augmented))
            
            # Add noise, ensuring positive values (e.g., Age/Quantity can't be negative)
            df_augmented[col] = (df_augmented[col] + noise).apply(lambda x: max(0, x))
            
            # Optionally, round to appropriate decimal places
            if 'Price' in col or 'Fee' in col:
                 df_augmented[col] = df_augmented[col].round(2)
            if 'Age' in col:
                 df_augmented[col] = df_augmented[col].round(0).astype(int)
            if 'Quantity' in col:
                 df_augmented[col] = df_augmented[col].round(0).astype(int).apply(lambda x: max(1, x))

    return df_augmented


def augment_csv(input_path, output_path, target_size, noise_level, numerical_cols, id_col):
    """
    Augments the CSV file to the target_size by sampling with replacement.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        create_dummy_csv(input_path)
        print("Please replace the content of 'raw_sales_data.csv' with your actual data and run the script again.")
        return

    try:
        # 1. Load the original data
        df_original = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    original_size = len(df_original)

    if original_size >= target_size:
        print(f"Original size ({original_size} rows) is already >= Target Size ({target_size} rows).")
        df_original.to_csv(output_path, index=False)
        print(f"Original data saved directly to '{output_path}'.")
        return

    num_to_add = target_size - original_size
    print(f"Original size: {original_size} rows.")
    print(f"Target size: {target_size} rows.")
    print(f"Generating {num_to_add} new rows...")

    # 2. Sample with replacement to preserve ratios
    df_augmented = df_original.sample(
        n=num_to_add, 
        replace=True, 
        random_state=42 # Set seed for reproducibility
    ).reset_index(drop=True)

    # 3. Handle CustomerID uniqueness
    # Find the maximum existing ID number (e.g., CUST0123 -> 123)
    try:
        # Extract numbers, handle NaN, find max. Assumes format like 'CUST####'
        max_id_num = df_original[id_col].str.extract(r'(\d+)').astype(float).max().iloc[0]
        start_id_num = int(max_id_num) + 1
    except Exception:
        # Fallback if ID column format is unexpected
        print("Warning: Could not parse CustomerID number sequence. Starting new IDs from 1.")
        start_id_num = 1 
        
    df_augmented = generate_unique_ids(df_augmented, start_id_num)

    # 4. Add slight random noise to numerical values
    df_augmented = add_random_noise(df_augmented, numerical_cols, noise_level)

    # 5. Combine and Save
    df_final = pd.concat([df_original, df_augmented], ignore_index=True)
    df_final = df_final.sample(frac=1).reset_index(drop=True) # Shuffle the final dataframe

    df_final.to_csv(output_path, index=False)
    
    print("-" * 40)
    print(f"SUCCESS! Data augmented from {original_size} to {len(df_final)} rows.")
    print(f"New dataset saved to: '{output_path}'")
    print("-" * 40)


if __name__ == '__main__':
    # You can change the parameters here or directly at the top of the file
    augment_csv(
        input_path=INPUT_FILE, 
        output_path=OUTPUT_FILE, 
        target_size=TARGET_SIZE, 
        noise_level=NOISE_LEVEL,
        numerical_cols=NUMERICAL_COLS,
        id_col=ID_COL
    )