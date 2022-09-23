import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

NUM_OF_PREV_ITEMS = 5

# we want to make
numpy.random.seed(1)

data_frame = read_csv('../Datasets/daily_min_temperatures.csv', usecols=[1])

# we just need the temperature column
data = data_frame._values

data = data.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

train, test = data[0:int(len(data) * 0.7), :], data[int(len(data) * 0.7): len(data), :]
train_x, train_y = reconstruct_data(train, NUM_OF_PREV_ITEMS)


def reconstruct_data(data_set, NUM_OF_PREV_ITEMS):
    x, y = [], []
    for i in range(len(data_set) - n - 1):
        a = data_set[i:(i + n), 0]
        x.append(a)
    return None


test_x, test_y = reconstruct_data(test, NUM_OF_PREV_ITEMS)

# reshape input to be [numOfSampels, time steps, numOfFeatures]
# time steps is 1 because we want to predict the next value (t+1)
# print((train_x.shape[0], 1, train_x.shape[1]))
train_x = numpy.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = numpy.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))


def reconstruct_data(data_set, n=1):