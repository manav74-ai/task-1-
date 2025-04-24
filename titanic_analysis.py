

import kagglehub
yasserh_titanic_dataset_path = kagglehub.dataset_download('yasserh/titanic-dataset')

print('Data source import complete.')



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

"""# Importing the dataset:"""

df=pd.read_csv("/kaggle/input/titanic-dataset/Titanic-Dataset.csv",index_col=0)

"""# Exploring the dataset:"""

df.head()

df.shape

df.info()

df.describe()

"""# Handling the missing values:"""

df.isnull().sum()

"""### Filling the missing value of Age column using median:"""

df['Age'].fillna(df['Age'].median(), inplace=True)

"""### Filling the missing value of Cabin column using most frequent:
-> because the Cabin column's datatype is object therefore we can use most_frequent method to fill the missing values.
"""

df['Cabin'].value_counts()

si=SimpleImputer(strategy='most_frequent')
df['Cabin'] = si.fit_transform(df[['Cabin']]).ravel()

"""### Filling the missing value of the Embarked columns:"""

df['Embarked'].value_counts()

df['Embarked']=si.fit_transform(df[['Embarked']]).ravel()

"""### Varifying the process:"""

df.isnull().sum()

"""# Splitting the cabin column:"""

df['Deck'] = df['Cabin'].str.extract(r'([A-Za-z])')  # Extract the letter part as Deck
df['Room'] = df['Cabin'].str.extract(r'(\d+)')      # Extract the number part as Room

df.head()

df.drop(columns=['Cabin'],inplace=True)

df.head()

df['Room']=df['Room'].astype('float')

"""# Converting categorical features into Numerical using encoding:"""

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['Deck']=le.fit_transform(df['Deck'])

df['Embarked']=le.fit_transform(df['Embarked'])

df['Sex']=le.fit_transform(df['Sex'])

df.head()

"""# Normalizing and standardizing the Numerical features:

### Applying the normalization technique on the Age and fare columns:
"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df['Normalized_Age']=scaler.fit_transform(df[['Age']])

df['Normalized_Fare'] = scaler.fit_transform(df[['Fare']])

df.head()

"""# visualizing the Outliers using boxplot and remove them:"""

# Visualizing outliers using boxplots
numerical_columns = ['Age', 'Fare', 'Room']  # Specify numerical columns to check for outliers
for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

# Handling outliers by capping them
for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

numerical_columns = ['Age', 'Fare', 'Room']
for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

"""-> As we can see there is no outlier in this numerical columns:"""

