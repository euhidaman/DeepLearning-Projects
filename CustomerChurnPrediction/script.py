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
