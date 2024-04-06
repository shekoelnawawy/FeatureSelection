import pandas as pd
import joblib
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from tsfresh import select_features

feature_set = 'Comprehensive' #'Minimal

features_2018 = joblib.load('./'+feature_set+'_2018.pkl')
features_2020 = joblib.load('./'+feature_set+'_2020.pkl')

df = pd.concat([features_2018, features_2020], ignore_index=True)

standardize = False
features = df # joblib.load('/Users/nawawy/Desktop/feature_engineering.pkl')
features.dropna(axis=1, inplace=True)
print(features)
y = pd.Series(['High', 'High', 'High', 'High', 'High', 'Low', 'High', 'Low', 'Low', 'High', 'High', 'High'])
print(y)
features_filtered = select_features(features, y)

print(features_filtered)