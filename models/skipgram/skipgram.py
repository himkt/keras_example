from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Reshape

from keras.utils import np_utils
import numpy as np


def example_generator(corpus, nb_vocab, window_size=3):
    nb_sequence = corpus.shape[0]

    for idx, center_word in enumerate(corpus):
        _from = idx - window_size
        _to = idx + window_size + 1

        center_words = list()
        contexts = list()

        for pos in range(_from, _to):
            if pos != idx and 0 <= pos < nb_sequence:
                center_words.append(center_word)
                contexts.append(corpus[pos])

        center_words = np.asarray(center_words, dtype=np.int)
        contexts = np_utils.to_categorical(contexts, nb_vocab)

        yield center_words, contexts


class Skipgram(Sequential):
    """
    skip-gram: predict contexts words given a word

    see: https://arxiv.org/pdf/1310.4546.pdf
    """

    def __init__(self, nb_vocab, nb_embedding):

        super(Skipgram, self).__init__()
        self.add(Embedding(
            input_dim=nb_vocab, output_dim=nb_embedding, input_length=1))
        self.add(Reshape((nb_embedding, )))
        self.add(Dense(nb_vocab, activation='softmax'))


if __name__ == '__main__':
    from chainer.datasets import get_ptb_words
    from chainer.datasets import get_ptb_words_vocabulary

    train_corpus, _, _ = get_ptb_words()

    word2idx = get_ptb_words_vocabulary()
    idx2word = {idx: word for word, idx in word2idx.items()}
    nb_vocab = max(word2idx.values()) + 1

    model = Skipgram(nb_vocab, 200)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
    model.summary()

    gen = example_generator(train_corpus, nb_vocab, window_size=3)
    for center, context in gen:
        model.train_on_batch(center, context)
        embeddings = model.get_weights()[0]

    print(embeddings)
