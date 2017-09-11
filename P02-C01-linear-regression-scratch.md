# Linear regression from scratch

Powerful ML libraries can eliminate repetitive work, but if you rely too much on
abstractions, you might never learn how neural networks really work under the
hood. So for this first example, let's get our hands dirty and build everything
from scratch, relying only on autograd and NDArray.

First, we'll import the same dependencies as in the [autograd
tutorial](/2-autograd.ipynb):

```{.python .input  n=1}
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
```

## Linear regression


We'll focus on the problem of linear regression. Given a collection of data
points ``X``, and corresponding target values ``y``, we'll try to find the line,
parameterized by a vector ``w`` and intercept ``b`` that approximately best
associates data points ``X[i]`` with their corresponding labels ``y[i]``. Using
some proper math notation, we want to learn a prediction $$\boldsymbol{\hat{y}}
= X \cdot \boldsymbol{w} + b$$
that minimizes the squared error across all examples $$\sum_{i=1}^n (\hat{y}_i-
y_i)^2.$$

You might notice that linear regression is an ancient model and wonder why we
would present a linear model as the first example in a tutorial series on neural
networks. Well it turns out that we can express linear regression as the
simplest possible (useful) neural network. A neural network is just a collection
of nodes (aka neurons) connected by directed edges. In most networks, we arrange
the nodes into layers with each taking input from the nodes below. To calculate
the value of any node, we first perform a weighted sum of the inputs (according
to weights ``w``) and then apply an *activation function*. For linear
regression, we have two layers, the input (depicted in orange) and a single
output node (depicted in green) and the activation function is just the identity
function.

In this picture, we visualize all of the components of each input as orange
circles.

![](https://raw.githubusercontent.com/zackchase/mxnet-the-straight-
dope/master/img/simple-net-linear.png)

To make things easy, we're going to work with a synthetic data where we know the
solution, by generating random data points ``X[i]`` and labels ``y[i] = 2 *
X[i][0]- 3.4 * X[i][1] + 4.2 + noise`` where the noise is drawn from a random
gaussian with mean ``0`` and variance ``.1``.

In mathematical notation we'd say that the true labeling function is
$$y = X \cdot w + b + \eta, \quad \text{for } \eta \sim
\mathcal{N}(0,\sigma^2)$$


```{.python .input  n=2}
X = nd.random_normal(shape=(10000,2))
y = 2* X[:,0] - 3.4 * X[:,1] + 4.2 + .01 * nd.random_normal(shape=(10000,))
```

Notice that each row in ``X`` consists of a 2-dimensional data point and that
each row in ``Y`` consists of a 1-dimensional target value.

```{.python .input  n=3}
print(X[0])
print(y[0])
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[-0.67765152  0.03629481]\n<NDArray 2 @cpu(0)>\n\n[ 2.74159384]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

We can confirm that for any randomly chosen point, a linear combination with the
(known) optimal parameters produces a prediction that is indeed close to the
target value

```{.python .input  n=4}
print(2 * X[0,0] - 3.4 * X[0,1] + 4.2)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 2.7212944]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

We can visualize the correspondence between our second feature (``X[:,1]``) and
the target values ``Y`` by generating a scatter plot with the Python plotting
package ``matplotlib``.

```{.python .input  n=5}
import matplotlib.pyplot as plt
plt.scatter(X[:,1].asnumpy(),y.asnumpy())
plt.show()
```

```{.json .output n=5}
[
 {
  "data": {
   "image/png": "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\nSol/B5cmQpqrzxKuAAAAAElFTkSuQmCC\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7fea77084630>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

## Data iterators

Once we start working with neural networks, we're going to need to iterate
through our data points quickly. We'll also want to be able to grab batches of
``k`` data points at a time, to shuffle our data. In MXNet, data iterators give
us a nice set of utilities for fetching and manipulating data. In particular,
we'll work with the simple  ``NDArrayIter`` class.

```{.python .input  n=6}
batch_size = 4
train_data = mx.io.NDArrayIter(X, y, batch_size, shuffle=True)
```

Once we've initialized our NDArrayIter (``train_data``), we can easily fetch
batches by calling ``train_data.next()``. ``batch.data`` gives us a list of
inputs. Because our model has only one input (``X``), we'll just be grabbing
``batch.data[0]``.

```{.python .input  n=7}
batch = train_data.next()
print(batch.data[0])
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.8431161  -0.65856469]\n [ 0.15113127 -0.16341539]\n [-0.04862006  2.09892035]\n [ 1.44590795 -1.54427004]]\n<NDArray 4x2 @cpu(0)>\n"
 }
]
```

We can also grab the corresponding labels

```{.python .input  n=8}
print(batch.label[0])
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[  4.7668643    5.07063389  -3.01890087  12.34088612]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

Finally, we can iterate over ``train_data`` just as though it were an ordinary
Python list:

```{.python .input  n=9}
counter = 0
train_data.reset()
for i, batch in enumerate(train_data):
    counter += 1
print(counter)
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "2500\n"
 }
]
```

You might notice that we called ``train_data.reset()`` before iterating through
it. This let's the iterator know to reshuffle the data, preparing for the next
pass. See what happens if we try to pass over the data again without first
hitting ``reset()``.

```{.python .input  n=10}
counter = 0
for i, batch in enumerate(train_data):
    counter += 1
print(counter)
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0\n"
 }
]
```

## Model parameters

Now let's allocate some memory for our parameters and set their initial values.

```{.python .input  n=11}
w = nd.random_normal(shape=(2,1))
b = nd.random_normal(shape=1)

params = [w, b]
```

In the succeeding cells, we're going to update these parameters to better fit
our data. That will involve taking the gradient (a multi-dimensional derivative)
of some *loss function* with respect to the parameters. We'll update each
parameter in the direction that reduces the loss. But first, let's just allocate
some memory for each gradient.

```{.python .input  n=12}
for param in params:
    param.attach_grad()
```

## Neural networks

Next we'll want to define our model. In this case, we'll be working with linear
models, the simplest possible *useful* neural network. To calculate the output
of the linear model, we simply multipy a given input with the model's weights
(``w``), and add the offset ``b``.

```{.python .input  n=13}
def net(X): 
    return nd.dot(X, w) + b
```

Ok, that was easy.

## Loss function

Train a model means making it better and better over the course of a period of
training. But in order for this goal to make any sense at all, we first need to
define what *better* means in the first place. In this case, we'll use the
squared distance between our prediction and the true value.

```{.python .input  n=14}
def square_loss(yhat, y): 
    return nd.mean((yhat - y) ** 2)
```

## Optimizer

It turns out that linear regression actually has a closed-form solution.
However, most interesting models that we'll care about cannot be solved
analytically. So we'll solve this problem by stochastic gradient descent. At
each step, we'll estimate the gradient of the loss with respect to our weights,
using one batch randomly drawn from our dataset. Then, we'll update our
parameters a small amount in the direction that reduces the loss. The size of
the step is determined by the *learning rate* ``lr``.

```{.python .input  n=15}
def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
```

## Execute training loop

Now that we have all the pieces all we have to do is wire them together by
writing a training loop. First we'll define ``epochs``, the number of passes to
make over the dataset. Then for each pass, we'll iterate through ``train_data``,
grabbing batches of examples and their corresponding labels.

For each batch, we'll go through the following ritual:
* Generate predictions (``yhat``) and the loss (``loss``) by executing a forward
pass through the network.
* Calculate gradients by making a backwards pass through the network
(``loss.backward()``).
* Update the model parameters by invoking our SGD optimizer.

```{.python .input  n=16}
epochs = 2
ctx = mx.cpu()
moving_loss = 0.

for e in range(epochs):
    train_data.reset()
    for i, batch in enumerate(train_data):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx).reshape((-1,1))
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, .001)
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        moving_loss = .99 * moving_loss + .01 * loss.asscalar()
            
        if (i + 1) % 500 == 0:
            print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i, moving_loss))    
```

```{.json .output n=16}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0, batch 499. Moving avg of loss: 7.71709577181\nEpoch 0, batch 999. Moving avg of loss: 1.04575801609\nEpoch 0, batch 1499. Moving avg of loss: 0.157409500179\nEpoch 0, batch 1999. Moving avg of loss: 0.0196059516681\nEpoch 0, batch 2499. Moving avg of loss: 0.00304722321276\nEpoch 1, batch 499. Moving avg of loss: 0.000488664000081\nEpoch 1, batch 999. Moving avg of loss: 0.000153205814944\nEpoch 1, batch 1499. Moving avg of loss: 0.000105157370772\nEpoch 1, batch 1999. Moving avg of loss: 9.53114924537e-05\nEpoch 1, batch 2499. Moving avg of loss: 9.79254770829e-05\n"
 }
]
```

## Conclusion

You've seen that using just mxnet.ndarray and mxnet.autograd, we can build
statistical models from scratch. In the following tutorials, we'll build on this
foundation, introducing the basic ideas between modern neural networks and
powerful abstractions in MXNet for building comples models with little code.

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
