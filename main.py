import pandas as pd
import joblib
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from tsfresh import select_features

feature_set = 'Comprehensive' #'Minimal

features_2018 = joblib.load('./'+feature_set+'_2018.pkl')
features_2020 = joblib.load('./'+feature_set+'_2020.pkl')

features = pd.concat([features_2018, features_2020], ignore_index=True)

standardize = False

features.dropna(axis=1, inplace=True)
# print(features)
y = pd.Series(['High', 'High', 'High', 'High', 'High', 'Low', 'High', 'Low', 'Low', 'High', 'High', 'High'])
# y = pd.Series([1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1])
# y = pd.Series([1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0])
# features_selected = select_features(features, y)
# print(features_selected)

X_full_train, X_full_test, y_train, y_test = train_test_split(features, y, test_size=.4)

classifier_full = DecisionTreeClassifier()
classifier_full.fit(X_full_train, y_train)
print(classification_report(y_test, classifier_full.predict(X_full_test)))