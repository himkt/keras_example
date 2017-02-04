Eve optimizer

Eve is optimizer based on Adam.

- [Adam](https://arxiv.org/abs/1412.6980)
- [Eve](https://arxiv.org/abs/1611.01505)


You can use Eve.

Download mnist examples from [here](https://raw.githubusercontent.com/pfnet/chainer/master/examples/mnist/train_mnist.py) and put on this directory.
Then edit L:60(check:2017/02/04) statement as bellow.

```python
# optimizer = chainer.optimizers.Adam()
from eve import Eve
optimizer = Eve()
```

And run ```python train_mnist.py```.

@article{koushik2016improving,
  title={Improving Stochastic Gradient Descent with Feedback},
  author={Koushik, Jayanth and Hayashi, Hiroaki},
  journal={arXiv preprint arXiv:1611.01505},
  year={2016}
}
