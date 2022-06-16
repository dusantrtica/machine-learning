import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('../Datasets/Datasets/credit_data.csv')

features = data[["income", "age", "loan"]]
target = data.default

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=30)

model = GaussianNB()
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)
