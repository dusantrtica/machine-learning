import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizer_v1 import SGD
from keras.optimizer_v2.gradient_descent import SGD
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# load data, 50k training samples and 10k test samples
# 32 x 32 pixel images - 10 output classes (labels)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize data set, one-hot encoding for the labels (1, 2, ...) will be replaced by 1s and 0s
# 0 = [1, 0, ... 0]
# 1 = [0, 1, ... 0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train / 255.0
X_test = X_test / 255.0

# construct CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

opt = SGD(lr=0.001, momentum=0.95)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2)
model_result = model.evaluate(X_test, y_test, verbose=0)

