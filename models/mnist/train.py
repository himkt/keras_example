from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.datasets import mnist

import numpy as np
from matplotlib import pyplot

pyplot.style.use('ggplot')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.0
shape = X_train.shape
X_train = X_train.reshape((shape[0], shape[1], shape[2], 1))
y_train = np_utils.to_categorical(y_train)

X_test = X_test.astype('float32') / 255.0
shape = X_test.shape
X_test = X_test.reshape((shape[0], shape[1], shape[2], 1))
y_test = np_utils.to_categorical(y_test)

nb_filter = 32
pool_size = (2, 2)
kernel_size = (3, 3)

model = Sequential()
model.add(Conv2D(nb_filter, kernel_size[0], kernel_size[1],
                 input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.summary()
model.fit(X_train, y_train,
          batch_size=32,
          nb_epoch=20)

y_pred = model.predict_classes(X_test)
y_true = np.argmax(y_test, axis=1)

X_miss = X_test[y_true != y_pred]
y_miss_true = y_true[y_true != y_pred]
y_miss_pred = y_pred[y_true != y_pred]

nb_mistake = X_miss.shape[0]
nb_column = 10
nb_row = nb_mistake // nb_column

for idx, data in enumerate(X_miss):
    y_p = y_miss_pred[idx]
    y_t = y_miss_true[idx]
    caption = '{}({})'.format(y_p, y_t)

    pyplot.subplot(nb_row + 1, nb_column, idx + 1)
    # pyplot.title(caption)
    pyplot.imshow(data.reshape((28, 28)))
    pyplot.tick_params(labelleft='off', labelbottom='off')
    pyplot.gray()

pyplot.savefig('result.pdf')
