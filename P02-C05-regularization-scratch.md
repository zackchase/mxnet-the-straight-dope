# Overfitting and regularization

In [the last tutorial](P02-C04-softmax-regression-tutorial.ipynb), we introduced
the task of multiclass classification. We showed how you can tackly this problem
with a linear model called logistic regression. Owing to some amount of
randomness, you might get slightly different results, but when I ran the
notebook, the model achieved 88.1% accuracy on the training data and actually
did slightly (but not significantly) better on the test data than on the data
the model has actually seen.

Not every algorithm that performs well on training data will also perform test
data. Take, for example, a trivial algorithm that memorizes its inputs and
stores the associated labels. This model would have 100% accuracy on training
data but would have no way of making any prediction at all on previously unseen
data.

The goal of machine learning is to produce models that *generalize* to
previously unseen data. When a model achieves low error on training data but not
test data, we sau that the model has *overfit*. This means that the model has
caught on to idiosyncratic features of the training data (e.g. one "2" happened
to have a white pixel in top-right corner), but hasn't really picked up on
general patterns.

Lots of factors govern whether a model will generalize well. And there's
actually a large body of mathematical analysis that guarantees the
generalization error for simple classes of models. *We won't get into this
theory but may delve deeper in a future chapter*.

On an intuitive level, a few factors tend to influence the generalizability of a
model class:
1. **The number of degrees of freedom.** When the number of tunable parameters
is large, models tend to be more susceptible to overfitting.
2. **The number of training examples.** It's trivially easy to overfit a dataset
containing only one or two examples even if your model is simple. But
overfitting a dataset with millions of examples requires an extremely flexible
model.

When classified handwritten digits before, we didn't overfit because our 60,000
training examples far out numbered the  $784 \times 10 = 7,840$ weights plus
$10$ bias terms gave us far fewer parameters than training examples. To

```{.python .input  n=43}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
ctx = mx.cpu()
mx.random.seed(1)
```

## Load the MNIST dataset

```{.python .input  n=45}
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

## Allocate model parameters and define model

```{.python .input  n=78}
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

```{.python .input  n=48}
def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat), axis=0, exclude=True)

def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
```

## Write evaluation loop to calculate accuracy

```{.python .input  n=50}
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

```{.python .input  n=52}
epochs = 1000
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
        SGD(params, .001)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        if i == 0:
            moving_loss = np.mean(loss.asscalar())
        else:
            moving_loss = .99 * moving_loss + .01 * np.mean(loss.asscalar())
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    if e % 100 == 99:
        print("Completed epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % 
              (e+1, moving_loss, train_accuracy, test_accuracy))       
```

```{.json .output n=52}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Completed epoch 100. Loss: 0.699882578467, Train_acc 0.883789, Test_acc 0.694336\nCompleted epoch 200. Loss: 0.383530223087, Train_acc 0.94043, Test_acc 0.726562\nCompleted epoch 300. Loss: 0.296192424847, Train_acc 0.974609, Test_acc 0.726562\nCompleted epoch 400. Loss: 0.219277466178, Train_acc 0.987305, Test_acc 0.734375\nCompleted epoch 500. Loss: 0.167076686896, Train_acc 0.996094, Test_acc 0.735352\nCompleted epoch 600. Loss: 0.129148652002, Train_acc 0.999023, Test_acc 0.738281\nCompleted epoch 700. Loss: 0.0998192830617, Train_acc 1.0, Test_acc 0.742188\nCompleted epoch 800. Loss: 0.0805371355478, Train_acc 1.0, Test_acc 0.743164\nCompleted epoch 900. Loss: 0.0675567659364, Train_acc 1.0, Test_acc 0.74707\nCompleted epoch 1000. Loss: 0.0582670229406, Train_acc 1.0, Test_acc 0.75\n"
 }
]
```

## Analysis

By the 700th epoch, our model achieves 100% accuracy on the training data.
However, it only classifies 75% of the test examples accurately.

[PLACEHOLDER]

## Regularization
[PLACEHOLDER]

```{.python .input  n=73}
def l2_penalty(params):
    penalty = nd.zeros(shape=1)
    for param in params:
        penalty = penalty + nd.sum(param ** 2)
    return penalty
```

## Re-initializing the parameters

[PLACEHOLDER]

```{.python .input  n=76}
for param in params:
    param[:] = nd.random_normal(shape=param.shape)
```

## Training L2-regularized logistic regression

[PLACEHOLDER]

```{.python .input  n=77}
epochs = 1000
moving_loss = 0.
l2_strength = .1

for e in range(epochs):
    train_data.reset()
    for i, batch in enumerate(train_data):
        data = batch.data[0].as_in_context(ctx).reshape((-1,784))
        label = batch.label[0].as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = nd.sum(cross_entropy(output, label_one_hot)) + l2_strength * l2_penalty(params)
        loss.backward()
        SGD(params, .001)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        if i == 0:
            moving_loss = np.mean(loss.asscalar())
        else:
            moving_loss = .99 * moving_loss + .01 * np.mean(loss.asscalar())
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    if e % 100 == 99:
        print("Completed epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % 
              (e+1, moving_loss, train_accuracy, test_accuracy))       
```

```{.json .output n=77}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Completed epoch 100. Loss: 417.827958109, Train_acc 0.904297, Test_acc 0.714844\nCompleted epoch 200. Loss: 222.459426625, Train_acc 0.962891, Test_acc 0.758789\nCompleted epoch 300. Loss: 123.724861786, Train_acc 0.984375, Test_acc 0.78125\nCompleted epoch 400. Loss: 72.6283419307, Train_acc 0.990234, Test_acc 0.806641\nCompleted epoch 500. Loss: 46.1087759935, Train_acc 0.991211, Test_acc 0.816406\nCompleted epoch 600. Loss: 32.333768166, Train_acc 0.992188, Test_acc 0.824219\nCompleted epoch 700. Loss: 25.1729688411, Train_acc 0.993164, Test_acc 0.827148\nCompleted epoch 800. Loss: 21.4474974968, Train_acc 0.992188, Test_acc 0.824219\nCompleted epoch 900. Loss: 19.5080785639, Train_acc 0.992188, Test_acc 0.824219\nCompleted epoch 1000. Loss: 18.4982315813, Train_acc 0.992188, Test_acc 0.829102\n"
 }
]
```

## Analysis

By adding L2 regularization we were able to increase the performance on test
data from 75% accuracy to 83% accuracy. That's a 32% reduction in error. In a
lot of applications, this big an improvement can make the difference between a
viable product and useless system.

[PLACEHOLDER FOR DISCUSSION]

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
