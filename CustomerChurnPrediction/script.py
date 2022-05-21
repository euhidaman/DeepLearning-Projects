# %%
import pandas as pd
from matplotlib import pyplot as pt
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
# creating a new dataset by saving non empty totalCharges column data
df1 = df[df.TotalCharges != ' ']
df1.shape

# %%
