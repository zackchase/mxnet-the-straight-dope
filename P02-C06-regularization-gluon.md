# Overfitting and regularization (with ``gluon``)

Now that we've built a [logistic regression model from
scratch](http://5-softmax-reression-scratch.ipynb), let's make this more
efficient with ``gluon``.

```{.python .input}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
ctx = mx.gpu(3)
```

## The MNIST Dataset

```{.python .input}
mnist = mx.test_utils.get_mnist()
num_examples = 1000
batch_size = 64
train_data = mx.io.NDArrayIter(
    mnist["train_data"][:num_examples], 
    mnist["train_label"][:num_examples], 
    batch_size, shuffle=True)
test_data = mx.io.NDArrayIter(
    mnist["test_data"][:num_examples], 
    mnist["test_label"][:num_examples], 
    batch_size, shuffle=True)
```

## Multiclass Logistic Regression

```{.python .input}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(10))
```

## Parameter initialization


```{.python .input}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Softmax Cross Entropy Loss

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 0.00})
```

## Evaluation Metric

```{.python .input}
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx).reshape((-1,784))
        label = batch.label[0].as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)        
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()
```

## Execute training loop

```{.python .input}
epochs = 1000
moving_loss = 0.

for e in range(epochs):
    train_data.reset()
    for i, batch in enumerate(train_data):
        data = batch.data[0].as_in_context(ctx).reshape((-1,784))
        label = batch.label[0].as_in_context(ctx)
        with autograd.record():
            output = net(data)
            cross_entropy = loss(output, label)
        cross_entropy.backward()
        trainer.step(data.shape[0])
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        if i == 0:
            moving_loss = nd.mean(cross_entropy).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(cross_entropy).asscalar()
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    if e % 100 == 99:
        print("Completed epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % 
              (e+1, moving_loss, train_accuracy, test_accuracy))           
```

## Conclusion

Now let's take a look at how to implement modern neural networks.

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}
params = net.collect_params()
```

```{.python .input}
params['sequential0_dense0_weight'].data
```

```{.python .input}

```

```{.python .input}

```
