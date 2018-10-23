'''

Keras network 테스트

'''

from keras.layers import Dense, LSTM

from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from keras import Sequential

import numpy as np
data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
def hold():
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, stateful=True,
                   batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True, stateful=True))
    model.add(LSTM(32, stateful=True))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # Generate dummy training data
    x_train = np.random.random((batch_size * 10, timesteps, data_dim))
    y_train = np.random.random((batch_size * 10, num_classes))

    print(x_train.shape)

    # Generate dummy validation data
    x_val = np.random.random((batch_size * 3, timesteps, data_dim))
    y_val = np.random.random((batch_size * 3, num_classes))


    model.summary()
    #
    # model.fit(x_train, y_train,
    #           batch_size=batch_size, epochs=5, shuffle=False,
    #           validation_data=(x_val, y_val))

    x_train = np.random.random((1000, 20))
    y_train = np.random.randint(2, size=(1000, 1))
    print(x_train.shape, y_train.shape)

    model = Sequential()
    # model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()


    # Generate dummy training data
    x_train = np.random.random((batch_size * 10, timesteps, data_dim))
    y_train = np.random.random((batch_size * 10, num_classes))

    print(x_train.shape)

    # Generate dummy validation data
    x_val = np.random.random((batch_size * 3, timesteps, data_dim))
    y_val = np.random.random((batch_size * 3, num_classes))


    max_caption_len = 16
    vocab_size = 10000


x_train = np.random.random((1000, 20))
print(np.shape(x_train))

x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

model = Sequential()
model.add(Dense(64, input_shape=(20), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


x_train = np.random.random((1, 20, 5))
y_train = np.random.random((1, 2))
print(np.shape(x_train))
model = Sequential()
model.add(TimeDistributed(Dense(64), input_shape=(20, 5)))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10))
model.add(Dense(2))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print(model.predict(x_train))


