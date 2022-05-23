# %%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.sample(5)

# %%
df.drop('customerID', axis='columns', inplace=True)
df.sample(5)

# %%
df.dtypes

# %%
# Converting the string data object to numeric value
pd.to_numeric(df['TotalCharges'], errors='coerce').isnull()

# %%
df[pd.to_numeric(df['TotalCharges'], errors='coerce').isnull()].shape

# %%
# creating a new dataset by removing empty totalCharges column data
df1 = df[df.TotalCharges != ' ']
df1.shape

# %%
# converting TotalCharges string into integer value
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
df1.TotalCharges.dtypes

# %%
tenure_churn_no = df1[df1.Churn == 'No'].tenure
tenure_churn_yes = df1[df1.Churn == 'Yes'].tenure

plt.xlabel('Tenure')
plt.ylabel('Number of Customers')
plt.title('Customer Churn Prediction Visualization')
plt.hist([tenure_churn_yes, tenure_churn_no], color=[
         'green', 'red'], label=['Churn=Yes', 'Churn=No'])
plt.legend()

# %%
# Function to print unique column values


def print_unique_col_values(df1):
    for column in df1:
        if df1[column].dtypes == 'object':
            print(f'{column} : {df1[column].unique()}')


# %%
# calling unique column values function
print_unique_col_values(df1)

df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)

# %%
print_unique_col_values(df1)

# %%
# Converting Yes and No, to 0 and 1
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

for col in yes_no_columns:
    df1[col].replace({'Yes': 1, 'No': 0}, inplace=True)

# %%
for col in df1:
    print(f'{col}: {df1[col].unique()}')

# %%
df1['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
df1.gender.unique()

# %%
# One hot encoding for multiple value columns
df2 = pd.get_dummies(data=df1, columns=[
                     'InternetService', 'Contract', 'PaymentMethod'])
df2.columns

# %%
# sample data check
df2.sample(3)

# %%
