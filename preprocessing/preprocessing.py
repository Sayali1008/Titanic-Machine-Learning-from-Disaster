import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Handling missing values
    df.loc[:, df.isnull().any(axis=0)]
    
    # Drop NaN rows for Embarked
    df.dropna(subset=['Embarked'])

    # Drop unnecessary columns
    df.drop(columns = ['Age', 'Cabin', 'PassengerId', 'Name', 'Ticket', 'Fare'], axis = 1, inplace = True)

    # One hot encode categorical data
    df = pd.get_dummies(df, columns = ['Sex'])
    df = pd.get_dummies(df, columns = ['Embarked'])

    # Shuffle the data
    df = df.sample(frac = 1)

    return df

def handle_imbalanced_data(df):
    class0 = df[df['Survived'] == 0][:342]
    class1 = df[df['Survived'] == 1]

    df_balanced = pd.concat([class0, class1])
    df_balanced = df_balanced.sample(frac = 1)

    return df_balanced

def generate_data(df_train, df_test):
    # Data preprocessing
    df_train = preprocess_data(df_train)
    df_train = handle_imbalanced_data(df_train)
    df_test = preprocess_data(df_test)

    # Split dataset into train, val and test
    X_df = df_train.iloc[:, df_train.columns != 'Survived']
    y_df = df_train['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
    y_train = np.reshape(y_train, (y_train.shape[0],))
    X_test = df_test.copy()

    return X_train, X_val, X_test, y_train, y_val
