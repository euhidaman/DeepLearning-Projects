# %%
# Importing Necessary libraries and modules
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# %%
# Reading and displaying data from the CSV file
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
# Scale large data values between 0 and 1
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

# %%
for col in df2:
    print(f'{col}: {df2[col].unique()}')

# %%
# Splitting data into, train and test data
X = df2.drop('Churn', axis='columns')
y = df2['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# %%
# Building the ANN model
model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=65)

# %%
# Evaluate model
model.evaluate(X_test, y_test)

# %%
# Predict using the model
yp = model.predict(X_test)
yp[:5]

# %%
y_test[:5]

# %%
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

y_pred[:10]

# %%
# Print classification report
print(classification_report(y_test, y_pred))

# %%
# Print confusion matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
