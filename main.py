import pandas as pd
import joblib
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tsfresh.feature_selection.relevance import calculate_relevance_table
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

feature_set = 'Comprehensive ' #'Minimal
standardize = False

features_2018 = joblib.load('./'+feature_set+'_2018.pkl')
features_2020 = joblib.load('./'+feature_set+'_2020.pkl')

features = pd.concat([features_2018, features_2020], ignore_index=True)

features.dropna(axis=1, inplace=True)
y = pd.Series(['High', 'High', 'High', 'High', 'High', 'Low', 'High', 'Low', 'Low', 'High', 'High', 'High'])

oversample = SMOTE(k_neighbors=2)
features, y = oversample.fit_resample(features, y)

relevance_table = calculate_relevance_table(features, y) #, fdr_level=5)
relevance_table = relevance_table[relevance_table["p_value"].notna()]
relevance_table.sort_values("p_value", inplace=True)
relevance_table.to_csv('./relevance.csv')

# relevance_table = relevance_table[relevance_table.relevant]
# print(relevance_table["feature"][:11])

# X_full_train, X_full_test, y_train, y_test = train_test_split(features, y, test_size=.4)
#
# classifier_full = DecisionTreeClassifier()
# classifier_full.fit(X_full_train, y_train)
# print(classification_report(y_test, classifier_full.predict(X_full_test)))

column_names = features.columns
if standardize:
    data = StandardScaler().fit_transform(features)
else:
    data = np.array(features)


seeds = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000]
for seed in seeds:
    clf = tree.DecisionTreeClassifier(random_state=seed)
    clf = clf.fit(data, y)
    tree.plot_tree(clf, feature_names=column_names[:-1], filled=True, class_names=['High', 'Low'], rounded = True)
    # plt.show()
    plt.savefig('./Trees/tree_'+str(seed), dpi=300)
