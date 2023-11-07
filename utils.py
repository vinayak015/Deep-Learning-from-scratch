import sklearn.datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np



def prepare_breast_cancer_binarised_data():
    breast_cancer = sklearn.datasets.load_breast_cancer()
    data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    data['class'] = breast_cancer.target
    X = data.drop('class', axis=1)
    Y = data['class']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

    '''Binarisation of input'''

    X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1, 0])
    X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[1, 0])

    # convert all to numpy arrays
    X_binarised_train = X_binarised_train.values
    X_binarised_test = X_binarised_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values

    return X_binarised_train, X_binarised_test, Y_train, Y_test


def prepare_breast_cancer_data():
    breast_cancer = sklearn.datasets.load_breast_cancer()
    data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    data['class'] = breast_cancer.target
    X = data.drop('class', axis=1)
    Y = data['class']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

    # convert all to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values

    return X_train, X_test, Y_train, Y_test


def prepare_mobile_dataset():
    data = pd.read_csv("mobile_cleaned.csv")
    X = data.drop("Rating", axis=1)
    Y = data["Rating"].values
    th = 4.2
    data['Class'] = (data['Rating'] >= th).astype("int")
    Y_binarised = data['Class'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, stratify=Y_binarised)
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    minmax_scaler = MinMaxScaler()
    Y_scaled_train = minmax_scaler.fit_transform(Y_train.reshape(-1, 1))
    Y_scaled_test = minmax_scaler.transform(Y_test.reshape(-1, 1))
    scaled_threshold = minmax_scaler.transform(np.array([th]).reshape(-1, 1))[0][0]
    Y_binarised_train = (Y_scaled_train > scaled_threshold).astype("int").ravel()
    Y_binarised_test = (Y_scaled_test > scaled_threshold).astype("int").ravel()
    return X_scaled_train, X_scaled_test, Y_scaled_train, Y_scaled_test, Y_binarised_train, Y_binarised_test
