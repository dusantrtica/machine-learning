import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# correlation is relationship between 2 variables
# can be >, =, < 0.
credit_data = pd.read_csv('../Datasets/Datasets/credit_data.csv')
print(credit_data.head())
print(credit_data.describe())
print(credit_data.corr())

# logistic regression with multiple features (income, age, loan)
# b0 + b1x1 + b2x2 + b3x3 (x1 = income, x2 = age, x3 = loan)
features = credit_data[['income', 'age', 'loan']]
target = credit_data.default

# 30% of the data set is for testing, 70% is for training
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)
model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)

# make predections for the feature_test
predictions = model.fit.predict(feature_test)

# confusion matrix describes the performance of a classification models
# * diagonal elements are the correct classifications
# * off-diagonals are the incorrect predictions
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
