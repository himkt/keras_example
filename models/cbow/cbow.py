from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Lambda
from keras.backend import mean

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


def example_generator(corpus, nb_vocab, window_size=3):
    nb_sequence = corpus.shape[0]

    for idx, center_word in enumerate(corpus):
        contexts = list()
        center_words = list()

        _from = idx - window_size
        _to = idx + window_size + 1

        context = []
        for pos in range(_from, _to):
            if pos != idx and 0 <= pos < nb_sequence:
                context.append(corpus[pos])

        contexts.append(context)
        center_words.append(center_word)

        contexts = pad_sequences(contexts, window_size*2)
        center_words = np_utils.to_categorical(center_words, nb_vocab)

        yield contexts, center_words


class CBoW(Sequential):
    """
    continuous bag-of-words: predict a word given contexts words

    see: https://arxiv.org/abs/1301.3781
    """

    def __init__(self, nb_vocab, nb_embedding, window_size):

        super(CBoW, self).__init__()
        self.add(Embedding(input_length=2*window_size,
                           input_dim=nb_vocab, output_dim=nb_embedding))
        self.add(Lambda(lambda x: mean(x, axis=1),
                        output_shape=(nb_embedding, )))
        self.add(Dense(nb_vocab, activation='softmax'))
