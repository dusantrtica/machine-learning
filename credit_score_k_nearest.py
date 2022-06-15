import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_csv('../Datasets/Datasets/credit_data.csv')

features = data[['income', 'age', 'loan']]
target = data.default

X = np.array(features).reshape(-1, 3)
y = np.array(target)

# X_new = X-min(X)/max(X)-min(X)
# normalization so that all values fall between 0 and 1
X = preprocessing.MinMaxScaler().fit_transform(X)
feature_train, feature_test, target_train, target_test = train_test_split(X, y, test_size=0.3)
# 20 closest neighbors are used in order to decide how to classify the item
model = KNeighborsClassifier(n_neighbors=20)
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

cross_valid_scores = []
for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
    cross_valid_scores.append(scores.mean())

print("Optimal K with cross-validation: ", np.argmax(cross_valid_scores))

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))


