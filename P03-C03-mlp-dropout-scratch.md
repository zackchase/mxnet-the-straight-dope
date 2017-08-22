# Dropout regularization from scratch

[**chapter status - very rough draft**]

If you're reading the tutorials in sequence, 
then you might remember from Part 2 
that machine learning models 
can be susceptible to overfitting. 
To recap: in machine learning,
our goal is to discover general patterns.
For example, we might want to learn as association between genetic markers
and the development of dementia in adulthood. 
Our hope would be to uncover a pattern that could be applied successfully to assess risk for the entire population.

However, when we train models, we don't have access to the entire population (or current or potential humans).
Instead, we can access only a small, finite sample.
Even in a large hospital system, we might get hundreds of thousands of medical records. 
Given such a finite sample size, it's possible to uncover spurious associations 
that don't hold up for unseen data.

Let's consider an extreme pathological case. 
Imagine that you want to learn to predict
which people will repay their loans. 
A lender hires you as a data scientist 
to investigate the case and gives you complete files on 100 applicants,
of which 5 defaulted on their loans within 3 years. 
The files might include hundreds of features 
including income, occupation, credit score, length of employment etcetera.
Imagine that they additionally give you video footage of their interview with a lending agent. 
That might seem like a lot of data! 

Now suppose that after generating an enormous set of features,
you discover that of the 5 applicants who defaults, 
all 5 were wearing blue shirts during their interviews,
while only 40% of general population wore blue shirts. 
Theirs a good chance that any model you train would pick up on this signal 
and use it as an important part of its learned pattern.

Even if defaulters are no more lkely to wear blue shirts, 
there's a 1% chance that we'll observe all five defaulters wearing blue shirts.
And keeping the sample size low we have hundreds or thousands of features,
we may observe a large number of spurious correlations. 
Given trillions of training examples, these false associations might disappear. 
But we seldom have that luxury.

The phenomena of fitting our training distribution more closely than the real distribution
is called *overfitting*, and the techniques used to combat overfitting are called regularization.
In the previous chapter, we introduced one classical approach to regularize statistical models. 
We penalized the size (the $\ell^2_2$ nor) of the weights, coercing them to take smaller values.
In probabilistic terms we might say this imposes a Gaussian prior on the value of the weights. 
But in more intuitive, functional terms, we can say this encourages the model to spead out its weights among many features and not to depend to much on a small number of potentially spurious associations. 
    

## With great flexibility comes overfitting liability

Given many more features than examples, linear models can overfit. 
But when there are many more examples than features, 
linear models can usually be counted on not to overfit.
Unfortunately this propensity to generalize well comes at a cost. 
For every feature, a linear model has to assign it either positive or negative weight.
Linear models can't take into account nuanced interactions between features.
In more formal texts, you'll see this phenomena discussed as the bias-variance tradeoff.
Linear models have high bias, (they can only represent a small class of functions),
but low variance (they give similar results across different random samples of the data).
[**point to more formal discussion of generalziation when chapter exists**]

Deep neural networks, however, occupy the opposite end of the bias-variance spectrum.
Neural networks are so flexible because they aren't confined to looking at each feature individually.
Instead, they can learn complex interactions among groups of features. 
For example, they might infer that "Nigeria" and "Wester Union" 
appearing together in an email indicates spam 
but that "Nigeria" without "Western Union" does not connote spam. 

Even for a small number of features, deep neural networks are capable of overfitting.
As one demonstration of the incredible flexibility of neural networks,
researchers showed that [neural networks perfectly classify randomly labeled data](https://arxiv.org/abs/1611.03530).
Let's think about what means. 
If the labels are assigned uniformly at random, and there are 10 classes, 
then no classifier can get better than 10% accuracy on holdout data. 
Yet even in these situations, when there is no true pattern to be learned, 
neural networks can perfectly fit the training labels. 

## Dropping out activations

In 2012, Professor Geoffrey Hinton and his students including Nitish Srivastava 
introduced a new idea for how to regularize neural network models. 
The inuition goes something like this. 
When a neural network overfits badly to training data,
each layer depends too heavily on the exact configuration
of features in the previous layer. 

To prevent the neural network from depending too much on any exact activation pathway,
Hinton and Srivastava proposed randomly *dropping out* (i.e. setting to $0$) 
the hidden nodes in every layer with probability $.5$.
Given a network with $n$ nodes we are sampling uniformly at random from the $2^n$ 
networks in which a subset of the nodes are turned off. 

![](./img/dropout.png)

One intuition here is that because the nodes to drop out are chosen randomly on every pass,
the representations in each layer can't depend on the exact values taken by nodes in the previous layer. 

## Making predictions with dropout models

However, when it comes time to make predictions, 
we want to use the full representational power of our model. 
In other words, we don't want to drop out activations at test time.
One principled way to justify the use of all nodes simultaneously,
despite not training in this fasion,
is that it's a form of model averaging. 
At each layer we average the representations of all of the $2^n$ dropout networks.
Because each node has a $.5$ probability of being on during training, 
its vote is scaled by $.5$ when we use all nodes at prediction time

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
ctx = mx.cpu()
```

## The MNIST dataset

Let's go ahead and grab our data.

[**SWITCH TO CIFAR TO GET BETTER FEEL FOR REGULARIZATION**]

```{.python .input  n=2}
mnist = mx.test_utils.get_mnist()
batch_size = 64
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
```

```{.python .input  n=3}
W1 = nd.random_normal(shape=(784,256), ctx=ctx) *.01
b1 = nd.random_normal(shape=256, ctx=ctx) * .01

W2 = nd.random_normal(shape=(256,128), ctx=ctx) *.01
b2 = nd.random_normal(shape=128, ctx=ctx) * .01

W3 = nd.random_normal(shape=(128,10), ctx=ctx) *.01
b3 = nd.random_normal(shape=10, ctx=ctx) *.01

params = [W1, b1, W2, b2, W3, b3]
```

Again, let's allocate space for gradients.

```{.python .input  n=4}
for param in params:
    param.attach_grad()
```

## Activation functions

If we compose a multi-layer network but use only linear operations, then our entire network will still be a linear function. That's because $\hat{y} = X \cdot W_1 \cdot W_2 \cdot W_2 = X \cdot W_4 $ for $W_4 = W_1 \cdot W_2 \cdot W3$. To give our model the capacity to capture nonlinear functions, we'll need to interleave our linear operations with activation functions. In this case, we'll use the rectified linear unit (ReLU):

```{.python .input  n=5}
def relu(X):
    return nd.maximum(X, 0)
```

## Dropout

```{.python .input  n=6}
def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    mask = nd.random_uniform(0, 1.0, X.shape, ctx=X.context) < keep_probability
    #############################
    #  Avoid division by 0 when scaling
    #############################
    if keep_probability > 0.0:
        scale = (1/keep_probability)
    else:
        scale = 0.0
    return mask * X * scale
```

```{.python .input  n=7}
A = nd.arange(20).reshape((5,4))
dropout(A, 0.0)
```

```{.python .input  n=8}
dropout(A, 0.5)
```

```{.python .input  n=9}
dropout(A, 1.0)
```

## Softmax output

```{.python .input  n=10}
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition
```

## The *softmax* cross-entropy loss function

```{.python .input  n=11}
def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)
```

## Define the model

Now we're ready to define our model

```{.python .input  n=12}
def net(X, drop_prob=0.0):
    #######################
    #  Compute the first hidden layer 
    #######################    
    h1_linear = nd.dot(X, W1) + b1
    h1 = relu(h1_linear)
    h1 = dropout(h1, drop_prob)
    
    #######################
    #  Compute the second hidden layer
    #######################
    h2_linear = nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)
    h2 = dropout(h2, drop_prob)
    
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

```{.python .input  n=13}
def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
```

## Evaluation metric

```{.python .input  n=14}
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

## Execute the training loop

```{.python .input  n=15}
epochs = 2
learning_rate = .001

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            ################################
            #   Drop out 50% of hidden activations on the forward pass
            ################################
            output = net(data, drop_prob=.5)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        if i == 0:
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %d. Loss: %f, Train_acc %f, Test_acc %f" % (e, moving_loss, train_accuracy, test_accuracy)) 
```

## Conclusion

Noice. With just two hidden layers containing 256 and 128 hidden nodes, repsectively, we can achieve over 95% accuracy on this task. 

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
