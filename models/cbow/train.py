from chainer.datasets import get_ptb_words
from chainer.datasets import get_ptb_words_vocabulary
from cbow import CBoW
from cbow import example_generator


if __name__ == '__main__':
    train_corpus, _, _ = get_ptb_words()

    word2idx = get_ptb_words_vocabulary()
    idx2word = {idx: word for word, idx in word2idx.items()}
    nb_vocab = max(word2idx.values()) + 1

    model = CBoW(nb_vocab, 200, 3)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
    model.summary()

    num_iter = 20

    for t in range(num_iter):
        loss = 0.0
        gen = example_generator(train_corpus, nb_vocab, window_size=3)

        for context, center in gen:
            loss += model.train_on_batch(context, center)

        print(loss)
