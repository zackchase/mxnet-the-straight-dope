# Overfitting and regularization (with ``gluon``)

Now that we've built a [logistic regression model from scratch](P02-C03-softmax-regression-scratch.ipynb), let's make this more efficient with ``gluon``.

[**ROUGH DRAFT - RELEASE STAGE: DOGFOOD**]

```{.python .input  n=9}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
ctx = mx.cpu()
```

## The MNIST Dataset

```{.python .input  n=10}
mnist = mx.test_utils.get_mnist()
num_examples = 1000
batch_size = 64
train_data = mx.gluon.data.DataLoader(
    mx.gluon.data.ArrayDataset(mnist["train_data"][:num_examples],
                               mnist["train_label"][:num_examples].astype(np.float32)), 
    batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(
    mx.gluon.data.ArrayDataset(mnist["test_data"][:num_examples],
                               mnist["test_label"][:num_examples].astype(np.float32)), 
    batch_size, shuffle=False)
```

## Multiclass Logistic Regression

```{.python .input  n=11}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(10))
```

## Parameter initialization

```{.python .input  n=12}
net.initialize(ctx=ctx)
```

## Softmax Cross Entropy Loss

```{.python .input  n=13}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

```{.python .input  n=14}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 0.00})
```

## Evaluation Metric

```{.python .input  n=15}
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        output = net(data)        
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()
```

## Execute training loop

```{.python .input  n=16}
epochs = 100
moving_loss = 0.

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            cross_entropy = loss(output, label)
        cross_entropy.backward()
        trainer.step(data.shape[0])
        
        #  Keep a moving average of the losses
        if i == 0:
            moving_loss = nd.mean(cross_entropy).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(cross_entropy).asscalar()
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    if e % 10 == 9:
        print("Completed epoch %4d. Loss: %f, Train_acc %f, Test_acc %f" % 
              (e+1, moving_loss, train_accuracy, test_accuracy))           
```

## Conclusion

Now let's take a look at how to implement modern neural networks. 

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
