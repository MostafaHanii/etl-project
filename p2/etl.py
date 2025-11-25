import pandas as pd
import numpy as np
import os
import io

# --- 1. EXTRACT: Data Loading ---

# IMPORTANT: This script assumes you have downloaded the CSV file and saved it 
# as 'raw_sales_data.csv' in the same directory.
input_filepath = 'augmented_sales_data.csv'

# Load the actual CSV file. We map column names and parse the date column.
df = pd.read_csv(
    input_filepath, 
    parse_dates=['OrderDate'],
    # Assuming the TotalPrice column may contain errors, we force it to numeric
    dtype={'TotalPrice': 'float64'}
)
print(f"Successfully loaded data from: {input_filepath}")

print("--- ETL Process Started ---")
print(f"Initial row count: {len(df)}")

# --- 2. TRANSFORM: Cleaning and Feature Engineering ---

# A. Data Type and Renaming
df.rename(columns={
    'OrderDate': 'InvoiceDate',
    'TotalPrice': 'Sales', # Use TotalPrice as the calculated Sales amount
    'ProductName': 'Description',
    'Region': 'Country'
}, inplace=True)

# Ensure Sales and numeric columns are correctly typed
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['ShippingFee'] = pd.to_numeric(df['ShippingFee'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').astype('Int64') # Use Int64 for nullable integer

# B. Handle Missing Values
# Drop rows where Sales or UnitPrice are null (core transaction data is required)
df.dropna(subset=['Sales', 'UnitPrice'], inplace=True)
print(f"Row count after dropping null Sales/UnitPrice: {len(df)}")

# Impute missing Age with the median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Fill missing ShippingFee with 0.00 (Assuming free shipping if not recorded)
df['ShippingFee'].fillna(0.00, inplace=True)

# Fill missing categorical fields with 'Unspecified'
df['Country'].fillna('Unspecified', inplace=True)

# C. Data Filtering and Business Logic
# Filter out transactions explicitly marked as 'Returned' or 'Cancelled' for Net Sales analysis
df = df[~df['ShippingStatus'].isin(['Returned', 'Cancelled'])]
print(f"Row count after filtering returns/cancellations: {len(df)}")

# D. Feature Engineering
# Create a Profit/Loss indicator (simplified: Sales - Cost of Goods Sold proxy)
# We assume Cost of Goods Sold (COGS) is 50% of UnitPrice for demonstration
df['COGS'] = df['UnitPrice'] * 0.5 * df['Quantity']
df['NetProfit'] = df['Sales'] - df['COGS'] - df['ShippingFee']

# Extract date components for time series analysis in Power BI
df['Date'] = df['InvoiceDate'].dt.date
df['Year'] = df['InvoiceDate'].dt.year
df['MonthName'] = df['InvoiceDate'].dt.strftime('%B')
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()

# E. Final Clean-up (Selecting final columns for Power BI)
final_columns = [
    'CustomerID', 'Gender', 'Country', 'Age', 'Description', 'Category', 
    'Quantity', 'UnitPrice', 'Sales', 'ShippingFee', 'NetProfit', 
    'ShippingStatus', 'InvoiceDate', 'Date', 'Year', 'MonthName', 'DayOfWeek'
]
df = df[final_columns]

print(f"Final clean row count: {len(df)}")

# --- 3. LOAD: Save the Clean Data ---
output_filepath = 'clean_sales_data.csv'
df.to_csv(output_filepath, index=False)
print(f"--- ETL Process Finished. Clean data saved to {output_filepath} ---")

# Print the first 5 rows of the clean data for verification
print("\nFirst 5 rows of the clean dataset:")
print(df.head())