# Multiclass logistic regression with ``gluon``

Now that we've built a [logistic regression model from
scratch](http://5-softmax-reression-scratch.ipynb), let's make this more
efficient with ``gluon``.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
```

We'll also want to set the compute context for our modeling. Feel free to go
ahead and change this to mx.gpu(0) if you're running on an appropriately endowed
machine.

```{.python .input  n=2}
ctx = mx.cpu()
```

## The MNIST Dataset

First, we'll grab the data.

```{.python .input  n=3}
mnist = mx.test_utils.get_mnist()
```

## Data Iterators

And load up two data iterators.

```{.python .input  n=4}
batch_size = 64
train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size, shuffle=True)
test_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size, shuffle=True)
```

We're also going to want to load up an iterator with *test* data. After we train
on the training dataset we're going to want to test our model on the test data.
Otherwise, for all we know, our model could be doing something stupid (or
treacherous?) like memorizing the training examples and regurgitating the labels
on command.

## Multiclass Logistic Regression

Now we're going to define our model.

```{.python .input  n=5}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(10))
```

## Parameter initialization


```{.python .input  n=6}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Softmax Cross Entropy Loss

Note, we didn't have to include the softmax layer because MXNet's has an
efficient function that simultaneously computes the softmax activation and
cross-entropy loss.

```{.python .input  n=7}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

And let's instantiate an optimizer to make our updates

```{.python .input  n=8}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## Evaluation Metric

This time, let's simplify the evaluation code by relying on MXNet's built-in
``metric`` package.

```{.python .input  n=9}
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

Because we initialized our model randomly, and because roughly one tenth of all
examples belong to each fo the ten classes, we should have an accuracy in the
ball park of .10.

```{.python .input  n=10}
evaluate_accuracy(test_data, net)
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "0.081707805"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Execute training loop

```{.python .input  n=11}
epochs = 10
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
            moving_loss = np.mean(cross_entropy.asnumpy()[0])
        else:
            moving_loss = .99 * moving_loss + .01 * np.mean(cross_entropy.asnumpy()[0])
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))    
    
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.502687432975, Train_acc 0.901153, Test_acc 0.907842\nEpoch 1. Loss: 0.474829963345, Train_acc 0.910848, Test_acc 0.915406\nEpoch 2. Loss: 0.462082007241, Train_acc 0.915278, Test_acc 0.917197\nEpoch 3. Loss: 0.453585840656, Train_acc 0.917944, Test_acc 0.918491\nEpoch 4. Loss: 0.447037873108, Train_acc 0.919493, Test_acc 0.919885\nEpoch 5. Loss: 0.441640226039, Train_acc 0.920659, Test_acc 0.921377\nEpoch 6. Loss: 0.437038913038, Train_acc 0.921658, Test_acc 0.921377\nEpoch 7. Loss: 0.433043815375, Train_acc 0.922458, Test_acc 0.922174\nEpoch 8. Loss: 0.429535013337, Train_acc 0.923291, Test_acc 0.922273\nEpoch 9. Loss: 0.426428586984, Train_acc 0.924157, Test_acc 0.922771\n"
 }
]
```

## Conclusion

Now let's take a look at how to implement modern neural networks.

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
