import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from preprocessing.preprocessing import generate_data
from logistic_regression import LogisticRegression as CustomLogisticRegression

def main():
    # Loading data
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")
    X_train, X_val, X_test, y_train, y_val = generate_data(df_train, df_test)

    ############### CustomLogisticRegression ###############
    
    # Model training
    model = CustomLogisticRegression()
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_val_pred = model.predict(X_val)
    y_val_acc = accuracy_score(y_val, y_val_pred)
    print(f"CustomLogisticRegression\tAccuracy: {y_val_acc}")

    # Model prediction
    y_test_pred = model.predict(X_test)

    ############### LogisticRegression ###############
    
    logReg = LogisticRegression()
    logReg.fit(X_train, y_train.ravel())
    y_val_pred2 = logReg.predict(X_val)
    y_val_acc2 = accuracy_score(y_val, y_val_pred2)
    print(f"LogisticRegression\tAccuracy: {y_val_acc2}")


    # Create submission file




if __name__ == '__main__':
    main()