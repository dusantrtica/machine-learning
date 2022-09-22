# from keras.models import Sequential
import pandas as pd
from keras.layers.core import Dense
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#from keras.optimizer_v1 import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
iris_data = load_iris()
features = iris_data.data

credit_data = pd.read_csv('../Datasets/Datasets/credit_data.csv')
features = credit_data[['income', 'age', 'loan']]
y = np.array(credit_data.default).reshape(-1, 1)

encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)
model = Sequential()
model.add(Dense(10, input_dim=3, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(train_features, test_targets, use_multiprocessing=True)
