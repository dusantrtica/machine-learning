import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# Keras hides the complexity of TensorFlow
# XOR is a non-linearly separable problem unlike AND, OR
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
target_data = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# we can define the lost function MSE or negative log likelihood
model.compile(loss='mean_squared_error',
              optimizer='adam', # how delta w are tuned
              metrics=['binary_accuracy']) # judge the performance of the model



# epoch is an iteration over the entire dataset
# verobose 0 is silent, 1 and 2 are showing results
model.fit(training_data, target_data, epochs=500, verbose=2)

# make predictions
print(model.predict(training_data).round())