# from keras.models import Sequential
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

# we have 3 targets, and 0, 1, 2 does not work for NN problems,
# we need to use OneHotEncoder to convert them to -1, 0, 1
labels = iris_data.target.reshape(-1, 1)

encoder = OneHotEncoder()
targets = encoder.fit_transform(labels).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# we can define the loss function MSE or negative log likelhood
# optimizer will find the right adjustement for the weights: SD, Adagrad, ADAM
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=10000, batch_size=20, verbose=2)
results = model.evaluate(test_features, test_targets, use_multiprocessing=True)
print('Traininig is finished... The lost and accuracy values are: ')
print(model.predict_proba(test_features))
