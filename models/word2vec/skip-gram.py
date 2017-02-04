import chainer
import iterator
import collections
from chainer.training import extensions


def convert(batch, device):
    center, context = batch
    if device >= 0:
        center = chainer.cuda.to_gpu(center)
        context = chainer.cuda.to_gpu(context)
    return center, context


class SkipGram(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__(
            embed=chainer.links.EmbedID(
                in_size=n_vocab, out_size=n_units
            ),
            loss_func=loss_func
        )

    def __call__(self, x, context):
        e = self.embed(context)
        shape = e.data.shape
        e = chainer.functions.reshape(e, (shape[0] * shape[1], shape[2]))

        # projection
        x = chainer.functions.broadcast_to(x[:, None], (shape[0], shape[1]))
        x = chainer.functions.reshape(x, (shape[0] * shape[1], ))

        # TODO: e.data.shape と x.data.shape が違うはずなのになぜこれで良いのか
        #       Continuous bag-of-words と loss_func を共有できる理由がよくわからない
        loss = self.loss_func(e, x)
        chainer.reporter.report({'loss': loss}, self)
        return loss


word2idx = chainer.datasets.get_ptb_words_vocabulary()
idx2word = {word: idx for idx, word in word2idx.items()}

train_corpus, test_corpus, _ = chainer.datasets.get_ptb_words()

count = collections.Counter(train_corpus)
count.update(collections.Counter(test_corpus))
cs = [count[w] for w in range(len(count))]

n_vocab = len(word2idx.keys())
n_negative = 5
n_window = 5
n_batch = 1000
n_units = 100

loss_func = chainer.links.NegativeSampling(n_units, cs, n_negative)

model = SkipGram(n_vocab, n_units, loss_func)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

gpu_id = -1
if gpu_id >= 0:
    chainer.cuda.get_device(gpu_id).use()
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(gpu_id).use()
    model.to_gpu()


train_iter = iterator.WindowIterator(train_corpus, n_window, n_batch)

updater = chainer.training.StandardUpdater(
    train_iter, optimizer, converter=convert, device=gpu_id)
trainer = chainer.training.Trainer(updater, (20, 'epoch'), out='result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss']))
trainer.extend(extensions.ProgressBar())

trainer.run()
