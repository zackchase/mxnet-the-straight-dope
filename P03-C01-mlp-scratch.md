# Multilayer perceptrons from scratch

Now that we've covered all the preliminaries, extending to deep neural networks is actually quite easy.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
ctx = mx.gpu()
```

## MNIST data (surprise!)

Let's go ahead and grab our data.

```{.python .input  n=2}
mnist = mx.test_utils.get_mnist()
num_inputs = 784
num_outputs = 10
batch_size = 64
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
```

## Multilayer perceptrons

Here's where things start to get interesting. Before, we mapped our inputs directly onto our outputs through a single linear transformation.

![](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/img/simple-softmax-net.png?raw=true)

This model is perfectly adequate when the underlying relationship between our data points and labels is approximately linear. When our data points and targets are characterized by a more complex relationship, a linear model will produce results with insufficient accuracy. We can model a more general class of functions by incorporating one or more *hidden layers*.

![](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/img/multilayer-perceptron.png?raw=true)

Here, each layer will require it's own set of parameters. To make things simple here, we'll assume two hidden layers of computation.

```{.python .input  n=3}
#######################
#  Set some constants so it's easy to modify the network later
####################### 
num_hidden = 256
weight_scale = .01

#######################
#  Allocate parameters for the first hidden layer
####################### 
W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=ctx)
b1 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=ctx)

#######################
#  Allocate parameters for the second hidden layer
####################### 
W2 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=ctx)
b2 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=ctx)

#######################
#  Allocate parameters for the output layer
####################### 
W3 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=ctx)
b3 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)

params = [W1, b1, W2, b2, W3, b3]
```

Again, let's allocate space for each parameter's gradients.

```{.python .input  n=4}
for param in params:
    param.attach_grad()
```

## Activation functions

If we compose a multi-layer network but use only linear operations, then our entire network will still be a linear function. That's because $\hat{y} = X \cdot W_1 \cdot W_2 \cdot W_2 = X \cdot W_4 $ for $W_4 = W_1 \cdot W_2 \cdot W3$. To give our model the capacity to capture nonlinear functions, we'll need to interleave our linear operations with activation functions. In this case, we'll use the rectified linear unit (ReLU):

```{.python .input  n=5}
def relu(X):
    return nd.maximum(X, nd.zeros_like(X))
```

## Softmax output

As with multiclass logistic regression, we'll want out outputs to be *stochastic*, meaning that they constitute a valid probability distribution. We'll use the same softmax activation function on our output to make sure that our outputs sum to one and are non-negative.

```{.python .input  n=6}
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition
```

## The *softmax* cross-entropy loss function

In the previous example, we calculate our model's output and then ran this output through the cross-entropy loss function: 

```{.python .input  n=7}
def cross_entropy(yhat, y):
    return - nd.nansum(y * nd.log(yhat), axis=0, exclude=True)
```

Mathematically, that's a perfectly reasonable thing to do. However, computationally, things can get hairy. We'll revist the issue at length in a chapter more dedicated to implementation and less interested in statistical modeling. But we're going to make a change here so we want to give you the gist of why.

When we calculate the softmax partition function, we take a sum of exponential functions:
$\sum_{i=1}^{n} e^{z_i}$. When we also calculate our numerators as exponential functions, then this can give rise to some big numbers in our intermediate calculations. The pairing of big numbers and low precision mathematics tends to make things go crazy. As a result, if we use our naive softmax implemenation, we might get horrific not a number (``nan``) results printed to secreen.

Our salvation is that even though we're computing these exponential functions, we ultimately plan to take their log in the cross-entropy functions. It turns out that by combining these two operators ``softmax`` and ``cross_entropy`` together, we can elude the numerical stability issues that might otherwise plague us during backpropagation. We'll want to keep the conventional softmax function handy in case we every want to evaluate the probabilities output by our model.

But instead of passing softmax probabilities into our loss function - we'll just pass our ``yhat_linear`` and compute the softmax and its log all at once inside the softmax_cross_entropy loss function simultaneously, which does smart things like the log-sum-exp trick ([see on Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)).


```{.python .input  n=8}
def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)
```

## Define the model

Now we're ready to define our model

```{.python .input  n=9}
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
    #  Compute the output layer.
    #  We will omit the softmax function here
    #  because it will be applied
    #  in the softmax_cross_entropy loss
    #######################
    yhat_linear = nd.dot(h2, W3) + b3
    return yhat_linear
```

## Optimizer

```{.python .input  n=10}
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
```

## Evaluation metric

```{.python .input  n=11}
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()
```

## Execute the training loop

```{.python .input  n=12}
epochs = 10
learning_rate = .001
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        
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

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.455451145229, Train_acc 0.885817, Test_acc 0.8897\nEpoch 1. Loss: 0.297250172348, Train_acc 0.921383, Test_acc 0.9205\nEpoch 2. Loss: 0.202016335186, Train_acc 0.946467, Test_acc 0.9451\nEpoch 3. Loss: 0.151867129294, Train_acc 0.960667, Test_acc 0.9584\nEpoch 4. Loss: 0.113816030109, Train_acc 0.9688, Test_acc 0.9637\nEpoch 5. Loss: 0.100374131216, Train_acc 0.97185, Test_acc 0.9658\nEpoch 6. Loss: 0.0873043180085, Train_acc 0.9779, Test_acc 0.9713\nEpoch 7. Loss: 0.0730908748383, Train_acc 0.98085, Test_acc 0.972\nEpoch 8. Loss: 0.068088298137, Train_acc 0.984883, Test_acc 0.9735\nEpoch 9. Loss: 0.0573755351742, Train_acc 0.986133, Test_acc 0.9731\n"
 }
]
```

## Conclusion

Nice! With just two hidden layers containing 256 hidden nodes, repsectively, we can achieve over 95% accuracy on this task. 

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
