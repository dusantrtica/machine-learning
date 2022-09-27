import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization  #
from keras.utils import np_utils
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# we can load the MNIST dataset from Keras datasets
# 60.000 training samples and 10.000 images in test set

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Tensorflow can handle format: (batch, height, width,channel)
features_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
features_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

# transform values within the range [0, 1]
features_test /= 255
features_train /= 255

# we have 10 output classes we want to end up with one hot
# encoding as we have seen for the Iris-dataset
# 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
targets_train = np_utils.to_categorical(y_train, 10)
targets_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
# input is 28x28 pxs image
# 32 is the number of filters - (3, 3) size of the filter
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
# normalizes the activation in the prev. layer after the convolutional phase
# transformation maintains the mean activation close to 0 std close to 1
# the scale of each dimension remains the same
# reduces running-time of training significantly












model.add(BatchNormalization())
model.add(Conv2D(3, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
# regularization helps to avoid overfitting
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax')) # categorical
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(features_train, targets_train, batch_size=128, epochs=2, validation_data=(features_test, targets_test),
          verbose=1)

score = model.evaluate(features_test, targets_test)
print('Test accuracy: %.2f' % score[1])
