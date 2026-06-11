import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load the dataset
df = pd.read_csv('vgsales.csv')

print("Original Data Shape:", df.shape)

# REQUIREMENT 1: Handle Missing Values
# The 'Year' column often has missing values. We fill it with the median.
df['Year'] = df['Year'].fillna(df['Year'].median())

# 'Publisher' is text (categorical). We fill missing ones with the most common publisher (mode).
df['Publisher'] = df['Publisher'].fillna(df['Publisher'].mode()[0])

# REQUIREMENT 2: Handle Categorical Variables
# Label Encoding for 'Platform' (turns platforms like PS4, PC into numbers like 0, 1)
label_encoder = LabelEncoder()
df['Platform_Encoded'] = label_encoder.fit_transform(df['Platform'])

# One-Hot Encoding for 'Genre' (creates separate columns for Action, Sports, etc.)
# drop_first=True prevents the dummy variable trap
df = pd.get_dummies(df, columns=['Genre'], drop_first=True)

# REQUIREMENT 3: Scale Numerical Features
# Sales numbers can vary wildly. We scale them so massive blockbusters don't skew future models.
scaler = StandardScaler()
numerical_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\nPreprocessing Complete. New Data Shape:", df.shape)
print("\nFirst 5 rows of cleaned data:")
print(df.head())