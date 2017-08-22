# Overfitting and regularization

In [the last tutorial](./P02-C03-softmax-regression-scratch.ipynb), we introduced the task of multiclass classification. We showed how you can tackle this problem with a linear model called logistic regression. Owing to some amount of randomness, you might get slightly different results, but when I ran the notebook, the model achieved 88.1% accuracy on the training data and actually did slightly (but not significantly) better on the test data than on the traning data. 
 
Not every algorithm that performs well on training data will also perform test data. Take, for example, a trivial algorithm that memorizes its inputs and stores the associated labels. This model would have 100% accuracy on training data but would have no way of making any prediction at all on previously unseen data. 

The goal of supervised learning is to produce models that *generalize* to previously unseen data. When a model achieves low error on training data but performs much worse on test data, we say that the model has *overfit*. This means that the model has caught on to idiosyncratic features of the training data (e.g. one "2" happened to have a white pixel in top-right corner), but hasn't really picked up on general patterns. 

We can express this more formally. The quantity we really care about is the test error $e$. Because this quantity reflects the error of our model when generalized previously unseen data, we commonly call it the *generalization error*. When we have simple models and abundant data, we expect the generalization error to resemble the training error. When we work with more complex models and fewer examples, we expect the training error to go down but the generalization gap to grow. Fixing the size of the dataset, the following graph should give you some intuition about what we generally expect to see.

![](img/regularization-overfitting.png)

What precisely constitutes model complexity is a complex matter. Many factors govern whether a model will generalize well. For example a model with more parameters might be considered more complex. A model whose parameters can take a wider range of values might be more comples. Often with neural networks, we think of a model that takes more training steps as more complex, and one subject to *early stopping* as less complex. 

It can be dificult to compare the complexity among members of very different model classes (say decision trees versus neural networks). Researchers in the field of statistical learning theory have developed a large body of mathematical analysis that formulizes the notion of model complexity and provides guarantees on the generalization error for simple classes of models. *We won't get into this theory but may delve deeper in a future chapter*.

To give you some intuition in this chapter, we'll focus on a few factors tend to influence the generalizability of a model class:
1. **The number of tunable parameters.** When the number of tunable parameters, sometimes denoted the number of degress of freedom, is large, models tend to be more susceptible to overfitting.
2. **The values taken by the parameters.** When weights can take a wider range of values, models can be more susceptible to over fitting.
3. **The number of training examples.** It's trivially easy to overfit a dataset containing only one or two examples even if your model is simple. But overfitting a dataset with millions of examples requires an extremely flexible model.


When classified handwritten digits before, we didn't overfit because our 60,000 training examples far out numbered the  $784 \times 10 = 7,840$ weights plus $10$ bias terms gave us far fewer parameters than training examples. Let's see how things can go wrong.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
ctx = mx.cpu()
mx.random.seed(1)
```

## Load the MNIST dataset

```{.python .input  n=2}
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

## Allocate model parameters and define model

```{.python .input  n=3}
W = nd.random_normal(shape=(784,10))
b = nd.random_normal(shape=10)

params = [W, b]

for param in params:
    param.attach_grad()
    
def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = nd.softmax(y_linear, axis=1)
    return yhat
```

## Define loss function and optimizer

```{.python .input  n=4}
def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat), axis=0, exclude=True)

def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
```

## Write evaluation loop to calculate accuracy

```{.python .input  n=5}
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

```{.python .input  n=6}
epochs = 100
moving_loss = 0.

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, .01)

        #  Keep a moving average of the losses
        if i == 0:
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    if e % 10 == 9:
        print("Completed epoch %4d. Loss: %f, Train_acc %f, Test_acc %f" % 
              (e+1, moving_loss, train_accuracy, test_accuracy))       
```

## What Happened?

By the 700th epoch, our model achieves 100% accuracy on the training data. However, it only classifies 75% of the test examples accurately. This is a clear case of overfitting. At a high level, let's reason about what went wrong. Because we have 7450 parameters and only 1000 data points, there are actually many settings of the parameters that could produce 100% accuracy on training data. 

To get some intuition imagine that we wanted to fit a dataset with 2 dimensional data and 2 data points. Our model has three degrees of freedom, and thus for any dataset can find an arbitrary number of separators that will perfectly caslsify our training points. Note below that we can produce completely orthogonal separators that both classify our training data perfectly. Even if it seems perposterous that they could both describe our training data well.

![](img/overfitting-low-data.png)

```{.python .input  n=7}
def l2_penalty(params):
    penalty = nd.zeros(shape=1)
    for param in params:
        penalty = penalty + nd.sum(param ** 2)
    return penalty
```

## Regularization

Now that we've characterized the problem of overfitting, we can begin talking about some solutions. 
Broadly speaking the family of techniques geared towards mitigating overfitting are referred to as *regularization*.
The core idea is this: when a model is overfitting, its training error is substantially lower than its test error. We're already doing as well as we possibly can on the training data, but our test data performance leaves something to be desired. Typically, regularization techniques attempt to trade of of our training performance in exchange for lowering our test error. 

There are several straight forward techniques we might employ. Given the intuition from the previous chart, we might attempt to make our model less complex. One way to do this would be to lower the number of free parameters. For example, we could throw away some subset of the our input features (and thus the correpsonding parameters) that we thought were least informative. 

![](img/regularization-overfitting.png)

Another approach is to limit the values that our weights might take. One common approach is to force the weights to take small values. 
[give more intuition with example of polynomial curve fitting]
We can accomplish this by changing our optimization objective to penalize the value of our weights. 
The most popular regularizer is the $\ell^2_2$ norm. For linear models, $\ell^2_2$ regularization as the additional benefit that it makes the solution unique, even when our model is overparametrized.
$$\sum_{i}(\hat{y}-y)^2 + \lambda || \textbf{w} ||^2_2$$
Here, $||\textbf{w}||$ is the $\ell^2_2$ norm and $\lambda$ is a hyper-parameter that determines how aggressively we want to push the weights towards 0.

In code, we can express the $\ell^2_2$ penalty succinctly:

## Re-initializing the parameters

```{.python .input  n=8}
for param in params:
    param[:] = nd.random_normal(shape=param.shape)
```

## Training L2-regularized logistic regression

```{.python .input  n=9}
epochs = 100
l2_strength = .1

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = nd.sum(cross_entropy(output, label_one_hot)) + l2_strength * l2_penalty(params)
        loss.backward()
        SGD(params, .01)

        #  Keep a moving average of the losses
        if i == 0:
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    if e % 10 == 9:
        print("Completed epoch %4d. Loss: %f, Train_acc %f, Test_acc %f" % 
              (e+1, moving_loss, train_accuracy, test_accuracy))       
```

## Analysis

By adding $L_2$ regularization we were able to increase the performance on test data from 75% accuracy to 83% accuracy. That's a 32% reduction in error. In a lot of applications, this big an improvement can make the difference between a viable product and useless system. Note that L2 regularization is just one of many ways of controlling capacity. Basically we assumed that small weight values are good. But there are many more ways to constrain the values of the weights:

* We could require that the total sum of the weights is small. That is what $L_1$ regularization does via the penalty $\sum_i |w_i|$. 
* We could require that the largest weight is not too large. This is what $L_\infty$ regularization does via the penalty $\max_i |w_i|$. 
* We could require that the number of nonzero weights is small, i.e. that the weight vectors are *sparse*. This is what the $L_0$ penalty does, i.e. $\sum_i I\{w_i \neq 0\}$. This penalty is quite difficult to deal with explicitly since it is nonsmooth. There is a lot of research that shows how to solve this problem approximatley using an $L_1$ penalty. 

![](img/regularization.png)

From left to right: $L_2$ regularization, which constrains the parameters to a ball, $L_1$ regularization, which constrains the parameters to a diamond (for lack of a better name, this is often referred to as an $L_1$-ball), and $L_\infty$ regularization, which constrains the parameters to a hypercube. 

All of this raises the question of **why** regularization is any good. After all, choice is good and giving our model more flexibility *ought* to be better (e.g. there are plenty of papers which show improvements on ImageNet using deeper networks). What is happening is somewhat more subtle. Allowing for many different parameter values allows our model to cherry pick a combination that is *just right* for all the training data it sees, without really learning the underlying mechanism. Since our observations are likely noisy, this means that we are trying to approximate the errors at least as much as we're learning what the relation between data and labels actually is. There is an entire field of statistics devoted to this issue - Computational Learning Theory. For now, a few simple rules of thumb suffice:

* Fewer parameters tend to be better than more parameters.
* Better engineering for a specific problem that takes the actual problem into account will lead to better models, due to the prior knowledge that data scientists have about the problem at hand.
* $L_2$ is easier to optimize for than $L_1$. In particular, many optimizers will not work well out of the box for $L_1$. Using the latter requires something called *proximal operators*.
* Dropout and other methods to make the model robust to perturbations in the data often work better than off-the-shelf $L_2$ regularization.

We conclude with an [XKCD Cartoon](https://xkcd.com/882/) which captures the entire situation more succinctly than the preceeding paragraph. 

![](https://imgs.xkcd.com/comics/significant.png)


For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
