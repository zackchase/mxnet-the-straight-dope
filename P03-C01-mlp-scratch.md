# Multilayer perceptrons from scratch

Now that we've covered all the preliminaries, extending to deep neural networks
is actually quite easy.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
ctx = mx.cpu()
```

## MNIST data (surprise!)

Let's go ahead and grab our data.

```{.python .input  n=2}
mnist = mx.test_utils.get_mnist()
batch_size = 64
train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size, shuffle=True)
test_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size, shuffle=True)
```

## Multilayer perceptrons

Here's where things start to get interesting. Before, we mapped our inputs
directly onto our outputs through a single linear transformation.

![](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/img/simple-
softmax-net.png?raw=true)

This model is perfectly adequate when the underlying relationship between our
data points and labels is approximately linear. When our data points and targets
are characterized by a more complex relationship, a linear model and produce
sucky results. We can model a more general class of functions by incorporating
one or more *hidden layers*.

![](https://github.com/zackchase/mxnet-the-straight-
dope/blob/master/img/multilayer-perceptron.png?raw=true)

Here, each layer will require it's own set of parameters. To make things simple
here, we'll assume two hidden layers of computation.

```{.python .input  n=3}
W1 = nd.random_normal(shape=(784,256)) *.01
b1 = nd.random_normal(shape=256) * .01

W2 = nd.random_normal(shape=(256,128)) *.01
b2 = nd.random_normal(shape=128) * .01

W3 = nd.random_normal(shape=(128,10)) *.01
b3 = nd.random_normal(shape=10) *.01

params = [W1, b1, W2, b2, W3, b3]
```

Again, let's allocate space for gradients.

```{.python .input  n=4}
for param in params:
    param.attach_grad()
```

## Activation functions

If we compose a multi-layer network but use only linear operations, then our
entire network will still be a linear function. That's because $\hat{y} = X
\cdot W_1 \cdot W_2 \cdot W_2 = X \cdot W_4 $ for $W_4 = W_1 \cdot W_2 \cdot
W3$. To give our model the capacity to capture nonlinear functions, we'll need
to interleave our linear operations with activation functions. In this case,
we'll use the rectified linear unit (ReLU):

```{.python .input  n=5}
def relu(X):
    return nd.maximum(X,nd.zeros_like(X))
```

## Softmax output

As with multiclass logistic regression, we'll want out outputs to be
*stochastic*, meaning that they constitute a valid probability distribution.
We'll use the same softmax activation functino on our output to make sure that
our outputs sum to one and are non-negative.

```{.python .input  n=6}
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition =nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition
```

## Define the model

Now we're ready to define our model

```{.python .input  n=7}
def net(X):
    #######################
    #  Compute the first hidden layer 
    #######################
    h1_linear = nd.dot(X, W1) + b1
    h1 = relu(h1_linear)
    
    #######################
    #  Compute the second hidden layer
    #######################
    h2_linear = nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)
    
    #######################
    #  Compute the output layer
    #######################
    yhat_linear = nd.dot(h2, W3) + b3
    yhat = softmax(yhat_linear)
    
    return yhat
```

## The cross-entropy loss function

Nothing changes here.

```{.python .input  n=8}
def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat), axis=0, exclude=True)
```

## Optimizer

```{.python .input  n=9}
def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
```

## Evaluation metric

```{.python .input  n=10}
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

## Execute the training loop

```{.python .input  n=11}
epochs = 5
moving_loss = 0.

for e in range(epochs):
    train_data.reset()
    for i, batch in enumerate(train_data):
        data = batch.data[0].as_in_context(ctx).reshape((-1,784))
        label = batch.label[0].as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, .01)
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        if i == 0:
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy)) 
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.172049476541, Train_acc 0.955674, Test_acc 0.953125\nEpoch 1. Loss: 0.104753504359, Train_acc 0.966518, Test_acc 0.962082\nEpoch 2. Loss: 0.0736007969344, Train_acc 0.977912, Test_acc 0.968153\nEpoch 3. Loss: 0.0588375955563, Train_acc 0.978045, Test_acc 0.96467\nEpoch 4. Loss: 0.0483494877147, Train_acc 0.986574, Test_acc 0.970641\n"
 }
]
```

## Conclusion

Noice. With just two hidden layers containing 256 and 128 hidden nodes,
repsectively, we can achieve over 95% accuracy on this task.

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
