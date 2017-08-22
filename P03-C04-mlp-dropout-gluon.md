# Implementing DropOut with ``gluon``

In [the previous chapter](./P03-C03-mlp-dropout-scratch.ipynb), 
we introduced DropOut regularization, implementing the algorithm from scratch. 
As a reminder, DropOut is a regularization technique 
that zeroes out some fraction of the nodes during training. 
Then at test time, we use all of the nodes, but scale down their values,
essentially averaging the various dropped out nets. 
If you're approaching the this chapter out of sequence,
and aren't sure how DropOut works, it's best to take a look at the implementation by hand
since ``gluon`` will manage the low-level details for us.

DropOut is a special kind of layer because it behaves differently 
when training and predicting. 
We've already seen how ``gluon`` can keep track of when to record vs not record the computation graph.
Since this is a ``gluon`` implementation chapter,
let's get intro the thick of things by importing our dependencies and some toy data.



```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from mxnet import gluon
ctx = mx.gpu()

```

## The MNIST dataset

```{.python .input  n=2}
batch_size = 64
num_inputs = 784
num_outputs = 10
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
```

## Define the model

Now we can add DropOut following each of our hidden layers. 

```{.python .input  n=3}
num_hidden = 256
net = gluon.nn.Sequential()
with net.name_scope():
    ###########################
    # Adding first hidden layer
    ###########################
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    ###########################
    # Adding dropout with rate .5 to the first hidden layer
    ###########################
    net.add(gluon.nn.Dropout(.5))
    
    ###########################
    # Adding first hidden layer
    ###########################
    net.add(gluon.nn.Dense(num_hidden, activation="relu")) 
    ###########################
    # Adding dropout with rate .5 to the second hidden layer
    ###########################
    net.add(gluon.nn.Dropout(.5))
    
    ###########################
    # Adding the output layer
    ###########################
    net.add(gluon.nn.Dense(num_outputs))
```

## Parameter initialization

Now that we've got an MLP with Dropout, let's register an initializer 
so we can play with some data.

```{.python .input  n=4}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Train mode and predict mode

Now that we have an MLP with DropOut, 
we can grab some data and pass it through the network.
We'll actually want to pass the example through the net twice,
just to see what effect DropOut is having on our predictions.

```{.python .input  n=5}
for x, _ in train_data:
    x = x.as_in_context(ctx)
    break
print(net(x[0:1]))
print(net(x[0:1]))
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.13859624  0.49433476  0.15499954 -0.068583    0.27810469 -0.10919482\n   0.05211569 -0.21952684  0.1023064  -0.33520821]]\n<NDArray 1x10 @gpu(0)>\n\n[[ 0.13859624  0.49433476  0.15499954 -0.068583    0.27810469 -0.10919482\n   0.05211569 -0.21952684  0.1023064  -0.33520821]]\n<NDArray 1x10 @gpu(0)>\n"
 }
]
```

Note that we got the exact same answer on both forward passes through the net!
That's because by, default, ``mxnet`` assumes that we are in predict mode. 
We can explicitly invoke this scope by placing code within a ``with autograd.predict_mode():`` block.

```{.python .input  n=6}
with autograd.predict_mode():
    print(net(x[0:1]))
    print(net(x[0:1]))
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.13859624  0.49433476  0.15499954 -0.068583    0.27810469 -0.10919482\n   0.05211569 -0.21952684  0.1023064  -0.33520821]]\n<NDArray 1x10 @gpu(0)>\n\n[[ 0.13859624  0.49433476  0.15499954 -0.068583    0.27810469 -0.10919482\n   0.05211569 -0.21952684  0.1023064  -0.33520821]]\n<NDArray 1x10 @gpu(0)>\n"
 }
]
```

Unless something's gone horribly wrong, you should see the same result as before. 
We can also run the code in *train mode*.
This tells MXNet to run our Blocks as they would run during training.

```{.python .input  n=7}
with autograd.train_mode():
    print(net(x[0:1]))
    print(net(x[0:1]))
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.33397433  0.59751952  0.46653447  0.32141876 -0.45945075 -0.08026656\n   0.34410784  0.20501006 -0.18988656  0.43075424]]\n<NDArray 1x10 @gpu(0)>\n\n[[ 0.58686924  0.8316716   0.32954857 -0.09099462  0.83973759 -0.7013377\n  -0.18787001 -0.56774801  0.7074067  -1.46942031]]\n<NDArray 1x10 @gpu(0)>\n"
 }
]
```

## Accessing ``is_training()`` status

You might wonder, how precisely do the Blocks determine 
whether they should run in train mode or predict mode?
Basically, autograd maintains a Boolean state 
that can be accessed via ``autograd.is_training()``. 
By default this falue is ``False`` in the global scope.
This way if someone just wants to make predictions and 
doesn't know anything about training models, everything will just work.
When we enter a ``train_mode()`` block, 
we create a scope in which ``is_training()`` returns ``True``. 

```{.python .input  n=8}
with autograd.predict_mode():
    print(autograd.is_training())
    
with autograd.train_mode():
    print(autograd.is_training())
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "False\nTrue\n"
 }
]
```

## Integration with ``autograd.record``

When we train, neural network models,
we nearly always enter ``record()`` blocks.
The purpose of ``record()`` is to build the computational graph.
And the purpose of ``train`` is to indicate that we are training our model.
These two are highly correlated but should not be confused.
For example, when we generate adversarial examples (a topic we'll investigate later)
we may want to record, but for the model to behave as in predict mode.
On the other hand, sometimes, even when we're not recording,
we still want to evaluate the model's training behavior.

A problem then arises. Since ``record()`` and ``train_mode()``
are distinct, how do we avoid having to declare two scopes every time we train the model?


```{.python .input  n=9}
##########################
#  Writing this every time could get cumbersome
##########################
with autograd.record():
    with autograd.train_mode():
        yhat = net(x)
```

To make our lives a little easier, record() takes one argument, ``train_mode``,
which has a default value of True.
So when we turn on autograd, this by default turns on train_mode
(``with autograd.record()`` is equivalent to
``with autograd.record(train_mode=True):``).
To change this default behavior
(as when generating adversarial examples),
we can optionally call record via
(``with autograd.record(train_mode=False):``).

## Softmax cross-entropy loss

```{.python .input  n=10}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

```{.python .input  n=11}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
```

## Evaluation metric

```{.python .input  n=12}
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
```

## Training loop

```{.python .input  n=13}
epochs = 10
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.328039098087, Train_acc 0.940866666667, Test_acc 0.9402\nEpoch 1. Loss: 0.250246155077, Train_acc 0.9511, Test_acc 0.9476\nEpoch 2. Loss: 0.194683401713, Train_acc 0.966616666667, Test_acc 0.9645\nEpoch 3. Loss: 0.173406506264, Train_acc 0.971116666667, Test_acc 0.9662\nEpoch 4. Loss: 0.163076896034, Train_acc 0.97655, Test_acc 0.9713\nEpoch 5. Loss: 0.134665723124, Train_acc 0.979566666667, Test_acc 0.9727\nEpoch 6. Loss: 0.140987931387, Train_acc 0.98215, Test_acc 0.9747\nEpoch 7. Loss: 0.12314310259, Train_acc 0.983183333333, Test_acc 0.9762\nEpoch 8. Loss: 0.118833732579, Train_acc 0.9843, Test_acc 0.9756\nEpoch 9. Loss: 0.115108108087, Train_acc 0.986533333333, Test_acc 0.975\n"
 }
]
```

## Conclusion

Now let's take a look at how to build convolutional neural networks.

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
