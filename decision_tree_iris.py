from sklearn import datasets
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris_data = datasets.load_iris()
features = iris_data.data
targets = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=20)

model = DecisionTreeClassifier(criterion='gini')
predicted = cross_validate(model, features, targets, cv=10)
print(np.mean(predicted['test_score']))
