# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

# Create a sample dataset
data = {'TransactionID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Amount': [100, 200, 150, 250, np.nan, 300, 350, 400, np.nan, 450],
        'IsFraud': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]}

df = pd.DataFrame(data)

# -------------------------------------
# Data Cleaning Techniques
# -------------------------------------

## 1. Handling missing values
df['Amount'].fillna(df['Amount'].mean(), inplace=True)

# -------------------------------------
# Data Transformation Techniques
# -------------------------------------

## 2. Feature scaling - Min-Max normalization
min_amount = df['Amount'].min()
max_amount = df['Amount'].max()
df['NormalizedAmount'] = (df['Amount'] - min_amount) / (max_amount - min_amount)

# -------------------------------------
# Data Integration Techniques
# -------------------------------------

## Create sample datasets from different sources
data_source1 = {'TransactionID': [1, 2, 3, 4, 5],
                'CustomerID': [101, 102, 103, 104, 105]}

data_source2 = {'TransactionID': [6, 7, 8, 9, 10],
                'CustomerID': [106, 107, 108, 109, 110]}

df_source1 = pd.DataFrame(data_source1)
df_source2 = pd.DataFrame(data_source2)

## Merge datasets from different sources
df_merged = pd.concat([df_source1, df_source2], ignore_index=True)

## Integrate merged dataset with the main dataset
df = pd.merge(df, df_merged, on='TransactionID')

# -------------------------------------
# Feature Engineering Section
# -------------------------------------

def feature_engineering(df):
    # Add your custom feature engineering functions here, e.g.:
    def create_interaction_terms(df, columns):
        for col1 in columns:
            for col2 in columns:
                if col1 != col2:
                    df[f"{col1}_{col2}"] = df[col1] * df[col2]
        return df

    # Apply the feature engineering functions to the dataset
    # E.g., create interaction terms for specific columns
    # df = create_interaction_terms(df, ['col1', 'col2', 'col3'])

    return df

df = feature_engineering(df)

# -------------------------------------
# Outlier Detection and Removal Section
# -------------------------------------

from scipy.stats import t, zscore

# Grubbs' test for outlier detection
def grubbs_test(data, alpha=0.05):
    N = len(data)
    Z = zscore(data)
    Z_max = np.abs(Z).max()
    t_alpha = t.ppf(1 - alpha / (2 * N), N - 2)
    G_calculated = Z_max / np.sqrt(1 + (t_alpha ** 2) / (N - 2 + t_alpha ** 2))
    G_critical = ((N - 1) * np.sqrt(np.square(t_alpha))) / (np.sqrt(N) * np.sqrt(N - 2 + np.square(t_alpha)))
    
    if G_calculated > G_critical:
        outlier_index = np.argmax(np.abs(Z))
        return outlier_index, True
    else:
        return None, False

def detect_and_remove_outliers(df, method='zscore', threshold=3.0):
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(df))
        outliers = np.where(z_scores > threshold)
        df = df[(z_scores < threshold).all(axis=1)]
    elif method == 'iqr':
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    elif method == 'gesd':
        for column in df.columns:
            has_outliers = True
            while has_outliers:
                index, has_outliers = grubbs_test(df[column])
                if has_outliers:
                    df = df.drop(index)
                    df = df.reset_index(drop=True)
    else:
        raise ValueError("Invalid outlier detection method. Use 'zscore', 'iqr', or 'gesd'.")

    return df

# Remove outliers using one of the methods: 'zscore', 'iqr', or 'gesd'
df = detect_and_remove_outliers(df, method='gesd')

# Display the cleaned dataset
print(df.head())

# Visualizations
## Histogram of transaction amounts
plt.hist(df['Amount'], bins=10, alpha=0.75, color='blue')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Transaction Amounts')
plt.show()

## Bar plot of fraud occurrences
fraud_counts = df['IsFraud'].value_counts()
plt.bar(fraud_counts.index, fraud_counts.values, alpha=0.75, color='green')
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.ylabel('Frequency')
plt.title('Fraud Occurrences')
plt.show()

# Plot the 'Amount' column before outlier removal
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Amount'], kde=True)
plt.title('Amount Distribution Before Outlier Removal')

# Remove outliers using one of the methods: 'zscore', 'iqr', or 'gesd'
df_cleaned = detect_and_remove_outliers(df, method='gesd')

# Plot the 'Amount' column after outlier removal
plt.subplot(1, 2, 2)
sns.histplot(df_cleaned['Amount'], kde=True)
plt.title('Amount Distribution After Outlier Removal')
plt.show()

# Display the cleaned dataset
print(df_cleaned.head())
