from chainer.datasets import get_ptb_words
from chainer.datasets import get_ptb_words_vocabulary
from skipgram import Skipgram
from skipgram import example_generator


if __name__ == '__main__':
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
