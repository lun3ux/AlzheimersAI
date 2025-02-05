import numpy as numpy
import matplotlib as matplotlib
import pandas as panda
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Numpy is for numerical operations.
# Pandas is for handling datasets. 
# Matplotlib is for data visualization. 
# Torch is for building and training.

panda.read_csv('/home/lune/datasets/alzheimers_prediction_dataset.csv')
df = panda.DataFrame(columns=["Country","Age","Gender", "Education Level", "BMI", "Physical Activity", "Smoking Status"])



m_gender_mappings = {'Male':0, 'Female':1}
m_physical_activity_mappings = {'Low': 0,'Medium': 1, 'High': 2}
m_smoking_statis_mappings = {'Never':0,'Former': 1,'Current': 3}
m_alcohol_consumption = {'Never': 0, 'Occasionally': 1, 'Regularly': 3}
m_hypertension_mappings = {'No': 0, "Yes":1}
m_colesterol_mappings = {'Low': 0, 'Normal': 1, 'High': 2}
m_diabetes_mappings = {'No': 0, 'Yes': 1}

for columns in df.columns:
    try:
        df[columns] = panda.to_numeric(df[columns])
    except ValueError:
        print("Chould not convert column {column}")

df = panda.read_csv('/home/lune/datasets/alzheimers_prediction_dataset.csv', skiprows=0)

df['Gender'] = df['Gender'].map(m_gender_mappings)
df['Physical Activity Level'] = df['Physical Activity Level'].map(m_physical_activity_mappings)
df['Smoking Status'] = df['Smoking Status'].map(m_smoking_statis_mappings)
df['Alcohol Consumption'] = df['Alcohol Consumption'].map(m_alcohol_consumption)
df['Diabetes'] = df['Diabetes'].map(m_diabetes_mappings)
df['Hypertension'] = df['Hypertension'].map(m_hypertension_mappings)
df['Cholesterol Level'] = df['Cholesterol Level'].map(m_colesterol_mappings)

df.describe()
df.info()
df.describe()
df.hist(figsize=(40,10))
plt.autoscale(enable=True)
plt.show()
torch.tensor(df)

le = LabelEncoder()
for columns in object_columns:
    df[columns] = le.fit_transform(df[columns])


# with open ('/home/lune/datasets/alzheimers_prediction_dataset.csv', 'r') as file:
#     print(file.read(500))
def normalize(x):
    return x/255.0


print(df.dtypes)
normalize(df)