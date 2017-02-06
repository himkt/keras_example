from keras.layers import Dense, Embedding
from keras.models import Sequential
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers.core import Activation


class FastText(Sequential):

    def __init__(self, nb_vocab, nb_embedding, nb_label):
        super(FastText, self).__init__()

        self.add(Embedding(input_dim=nb_vocab, output_dim=nb_embedding))
        self.add(Lambda(
            lambda x: K.mean(x, axis=1),
            output_shape=(nb_embedding, )
        ))
        self.add(Dense(nb_class))
        self.add(Activation('softmax'))


if __name__ == '__main__':

    from keras.datasets import imdb
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils.np_utils import to_categorical

    nb_vocab = 5000
    maxlen = 400
    nb_class = 2
    nb_embedding = 50

    (train_X, train_y), (test_X, test_y) = imdb.load_data(
        nb_words=nb_vocab, maxlen=maxlen)

    train_X = pad_sequences(train_X, maxlen)
    train_y = to_categorical(train_y, nb_classes=nb_class)

    test_X = pad_sequences(test_X, maxlen)
    test_y = to_categorical(test_y, nb_classes=nb_class)

    model = FastText(nb_vocab, nb_embedding, nb_class)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    model.fit(train_X, train_y,
              nb_epoch=10, batch_size=32,
              validation_data=(test_X, test_y))
