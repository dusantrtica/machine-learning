import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from tensorflow.keras.optimizers import Adam

dataset = load_iris()
features = dataset.data

# we need array of arrays, for NN training
# that is why we need to reshape it from one dim into 2 dim array
y = dataset.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse=False)
# activation functions need
targets = encoder.fit_transform(y)
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)


model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
# for the classification we use soft max fn as it assigns probabilty to each of the class (output)
model.add(Dense(3, activation='softmax'))

optimizer = Adam(lr=0.005)
model.compile(optimizer=optimizer, metrics=['accuracy'], loss='categorical_crossentropy')
model.fit(train_features, train_targets, epochs=1000, batch_size=20, verbose=2)
results = model.evaluate(test_features, test_targets)
print("Accuracy on the test dataset: %.2f" % results[1])