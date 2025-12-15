# Chronic Disease Prediction – Machine Learning Code

## 1. Import Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import sklearn

2.Load dataset
 
df_data = pd.read_csv('kidney_disease.csv')
df_data.shape
df_data.info()
df_data.head()

3.Drop irrelevant column

df_data.drop('id', axis=1, inplace=True)
df_data.describe()

4.Rename columns

df_data.columns = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
    'potassium', 'haemoglobin', 'packed_cell_volume',
    'white_blood_cell_count', 'red_blood_cell_count', 'hypertension',
    'diabetes_mellitus', 'coronary_artery_disease', 'appetite',
    'peda_edema', 'anemia', 'class'
]

5.Convert text columns to Numeric

text_columns = [
    'packed_cell_volume',
    'white_blood_cell_count',
    'red_blood_cell_count'
]

for i in text_columns:
    print(f"{i} : {df_data[i].dtype}")

def convert_text_to_numeric(df_data, column):
    df_data[column] = pd.to_numeric(df_data[column], errors='coerce')

for column in text_columns:
    convert_text_to_numeric(df_data, column)
    print(f"{column} : {df_data[column].dtype}")

6.Missing values Analysis

missing = df_data.isnull().sum()
missing[missing > 0].sort_values(ascending=False).head(20)

7.Missing values imputation

def mean_value_imputation(df_data, column):
    df_data[column].fillna(df_data[column].mean(), inplace=True)

def mode_value_imputation(df_data, column):
    df_data[column].fillna(df_data[column].mode()[0], inplace=True)

num_cols = [col for col in df_data.columns if df_data[col].dtype != 'object']
for col_name in num_cols:
    mean_value_imputation(df_data, col_name)

cat_cols = [col for col in df_data.columns if df_data[col].dtype == 'object']
for col_name in cat_cols:
    mode_value_imputation(df_data, col_name)

8.Clean categorical inconsistence

df_data['diabetes_mellitus'] = df_data['diabetes_mellitus'].replace(
    {' yes':'yes', '\tno':'no', '\tyes':'yes'}
)

df_data['coronary_artery_disease'] = df_data['coronary_artery_disease'].replace(
    '\tno', 'no'
)

df_data['class'] = df_data['class'].replace(
    {'ckd\t':'ckd', 'notckd':'not ckd'}
)

9.Encode categorical variables

df_data['class'] = df_data['class'].map({'ckd':1, 'not ckd':0})
df_data['red_blood_cells'] = df_data['red_blood_cells'].map({'normal':1,'abnormal':0})
df_data['pus_cell'] = df_data['pus_cell'].map({'normal':1,'abnormal':0})
df_data['pus_cell_clumps'] = df_data['pus_cell_clumps'].map({'present':1,'notpresent':0})
df_data['bacteria'] = df_data['bacteria'].map({'present':1,'notpresent':0})
df_data['hypertension'] = df_data['hypertension'].map({'yes':1,'no':0})
df_data['diabetes_mellitus'] = df_data['diabetes_mellitus'].map({'yes':1,'no':0})
df_data['coronary_artery_disease'] = df_data['coronary_artery_disease'].map({'yes':1,'no':0})
df_data['appetite'] = df_data['appetite'].map({'good':1,'poor':0})
df_data['peda_edema'] = df_data['peda_edema'].map({'yes':1,'no':0})
df_data['anemia'] = df_data['anemia'].map({'yes':1,'no':0})

10.Correlation Heatmap

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,8))
sns.heatmap(df_data.corr(), annot=True, linewidths=0.5)
plt.show()

11.Feature correlation with target

target_corr = df_data.corr()['class'].abs().sort_values(ascending=False)[1:]
target_corr

12.Train-Test split

from sklearn.model_selection import train_test_split

X = df_data.drop("class", axis=1)
y = df_data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=25
)

13.Model initialisation

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = []
models.append(('Naive Bayes', GaussianNB()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=8)))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('SVM', SVC(kernel='linear')))

14.Model training and Evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(name)
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print()



