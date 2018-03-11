import pandas as pd
from datetime import datetime
import os
from os import path
import sys

def parse(x):
    return datetime.strptime(x, "%Y %m %d %H")

os.listdir(".")

cur_dir = path.dirname(__file__)
sys.path.append(cur_dir)

# load data and format date
dataset = pd.read_csv('./PRSA_data_2010.1.1-2014.12.31.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.head()

# remove index column
dataset.drop('No', axis=1, inplace=True)
dataset.head()

# rename columns' name
dataset.columns = ['pollution', 'dewp', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
dataset.head()

# rename columns' name
dataset.columns = ['pollution', 'dewp', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
dataset.head()

# deal with Na values
dataset['pollution'].fillna(0, inplace=True)
dataset.head(5)

dataset = dataset[24:]
dataset.head()

# learn MinMaxScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
x_train = np.array([[1,-1,2],[2,0,0],[0,1,-1]])
min_max_scaler = MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
print(x_train)
print(x_train_minmax)
print(range(1,10))

# draw each feature
import matplotlib.pyplot as plt
values = dataset.values
print(values)
#print values.shape[1]
groups = [0,1,2,3,5,6,7]
i = 1
fig = plt.figure()
n_ax = []
for group in groups:
    ax = fig.add_subplot(len(groups), 1, i)
    ax.set_title(dataset.columns[group], y=0.5, loc='right')
    ax.plot(values[:,group])
    n_ax.append(ax)
    i += 1
    #plt.subplot(len(groups), 1, i)
    #plt.plot(values[:, group])
    #plt.title(dataset.columns[group], y=0.5, loc='right')
fig.suptitle("info")
plt.show()

# encode label feature
from sklearn.preprocessing import LabelEncoder
print(values[:,4])
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])

pd.set_option('display.width', 300)
print(values[:, 4][:100])
print(dataset.head(10))

# 最小最大值标准化
values = values.astype('float32')
scaler = MinMaxScaler()
scalerd = scaler.fit_transform(values)
print(scalerd)

print(type(scalerd))
if type(scalerd) is list:
    print('zzzzz')
df = pd.DataFrame(scalerd)
df.head()

cols, names = list(), list()
print(scalerd.shape)
n_vars = 1 if type(scalerd) is list else scalerd.shape[1]
n_in = 1
n_out = 1
dropnan = True
for i in range(n_in, 0, -1):
    print(i)
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
print(names)
for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
print(names)
print(cols)

from pandas import concat
#print cols
agg = concat(cols,axis=1)
#print agg
agg.columns = names
agg.dropna(inplace=True)
print(agg)

reframed = agg
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
reframed.head()

# split train set and test set
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
print(train_X)
print(train_X.shape)
print(train_y)
print(train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print(train_X)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=10, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

data1 = np.array([0.149899,0.470588,0.229508,0.672728,0.66666,0.074853,0.000000,0.0]).reshape((1, 1,8))
print(data1)
model.predict(data1)