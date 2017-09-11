# Multiclass logistic regression from scratch

If you've made it through our tutorial on linear regression from scratch, then
you're past the hardest part. You already know how to load and manipulate data,
build computation graphs on the fly, and take derivatives. You also know how to
define a loss function, construct a model, and write your own optimizer.

Nearly all neural networks that we'll build in the real world consist of these
same fundamental parts. The main differences will be the type and scale of the
data, and the complexity of the models. And every year or two, a new hipster
optimizer comes around, but at their core they're all subtle variations of
stochastic gradient descent.

So let's work on a more interesting problem nowh. We're going to classify images
of handwritten digits like these:
![png](https://raw.githubusercontent.com/dmlc/web-
data/master/mxnet/example/mnist.png)
We're going to implemment a model called multiclass logistic regression. Other
common names for this model include softmax regression and multinomial
regression. To start, let's import our bag of libraries.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
```

We'll also want to set the compute context for our modeling. Feel free to go
ahead and change this to mx.gpu(0) if you're running on an appropriately endowed
machine.

```{.python .input  n=2}
ctx = mx.cpu()
```

## The MNIST dataset

This time we're going to work with real data, each a 28 by 28 centrally cropped
black & white photograph of handwritten digit. Our task will be come up with a
model that can associate each image with the digit (0-9) that it depicts.

To start, we'll use MXNet's utility for grabbing a copy of this dataset.

```{.python .input  n=3}
mnist = mx.test_utils.get_mnist()
```

The retrieved object consists of a Python dictionary with four keys:
``train_data``,  ``train-label``, ``test_data``, and ``test_label``. We can take
a look at the format of examples.

```{.python .input  n=4}
image = mnist["train_data"][0]
print(image.shape)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(1, 28, 28)\n"
 }
]
```

Note that each image has been formatted as a 3-tuple (channel, height, width).
For color images, the channel would have 3 dimensions (red, green and blue). We
can also check out the labels.

```{.python .input  n=5}
label = mnist["train_label"][0]
print(label)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "5\n"
 }
]
```

Machine learning libraries generally expect to find images in (batch, channel,
height, width) format. However, most libraries for visualization prefer (height,
width, channel). Let's transpose our image into the expected shape.

```{.python .input  n=6}
im = np.tile(image.transpose(1,2,0), (1,1,3))
print(im.shape)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(28, 28, 3)\n"
 }
]
```

Now we can visualize our image and make sure that our data and labels line up.

```{.python .input  n=7}
import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()
```

```{.json .output n=7}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f6ed21b26a0>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

Ok, that's a beautiful five.

## Load the data iterator

Now let's load these images into data iterator so we don't have to do the heavy
lifting.

```{.python .input  n=8}
batch_size = 64
train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size, shuffle=True)
```

We're also going to want to load up an iterator with *test* data. After we train
on the training dataset we're going to want to test our model on the test data.
Otherwise, for all we know, our model could be doing something stupid (or
treacherous?) like memorizing the training examples and regurgitating the labels
on command.

```{.python .input  n=9}
test_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size, shuffle=True)
```

## Allocate model parameters

Now we're going to define our model. For this example, we're going to ignore the
multimodal structure of our data and just flatten each image into a single 1D
vector with 28x28 = 784 components.

Because our task is multiclass classification, we want to assign a probability
to each of the classes P(Y=c|X) given the input X. In order to do this we're
going to need one vector of 784 weights for each class, connecting each feature
to the corresponding output. Because there are 10 classes, we can collect these
weights together in a 784 by 10 matrix.

We'll also want to allocate one offset for each of the outputs. We call these
offsets the *bias term* and collect them in the 10-dimensional array ``b``.

```{.python .input  n=10}
W = nd.random_normal(shape=(784,10))
b = nd.random_normal(shape=10)

params = [W, b]
```

As before, we need to let MXNet know that we'll be expecing gradients
corresponding to each of these parameters during training.

```{.python .input  n=11}
for param in params:
    param.attach_grad()
```

## Multiclass logistic regression

In linear regression tutorial, we performed regression, so we had just one
output *yhat* and tried to push this value as close as possible to the true
target *y*. Here, instead of regression, we are performing *classification*,
where we want to assign each input *X* to one of *L* classes.

The basic modeling idea is that we're going to linearly map our input *X* onto
10 different real valued outputs ``y_linear``. Then before, outputting these
values, we'll want to normalize them so that they are non-negative and sum to 1.
This normalization allows us to interpret the output yhat as a valid probability
distribution.



```{.python .input  n=12}
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition =nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition
```

```{.python .input  n=13}
sample_y_linear = nd.random_normal(shape=(2,10))
sample_yhat = softmax(sample_y_linear)
print(sample_yhat)
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.01466005  0.03104205  0.09487285  0.11615293  0.07316667  0.01516553\n   0.44094777  0.08199082  0.0917872   0.04021411]\n [ 0.0309542   0.07588483  0.37230074  0.03313261  0.0499984   0.13276106\n   0.14566724  0.02354518  0.08515968  0.05059606]]\n<NDArray 2x10 @cpu(0)>\n"
 }
]
```

Let's confirm that indeed all of our rows sum to 1.

```{.python .input  n=14}
print(nd.sum(sample_yhat, axis=1))
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 1.  1.]\n<NDArray 2 @cpu(0)>\n"
 }
]
```

But for small rounding errors, the function works as expected.

## Define the model

Now we're ready to define our model

```{.python .input  n=15}
def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = softmax(y_linear)
    return yhat
```

## The  cross-entropy loss function

Before we can start training, we're going to need to define a loss function that
makes sense when our prediction is a  probability distribution.

The relevant loss function here is called cross-entropy and it may be the most
common loss function you'll find in all of deep learning. That's because at the
moment, classification problems tend to be far more abundant than regression
problems.

The basic idea is that we're going to take a target Y that has been formatted as
a one-hot vector, meaning one value corrsponding to the correct label is set to
1 and the others are set to 0, e.g. ``[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]``.


The basic idea of cross-entropy loss is that we only care about how much
probability the prediction assigned to the correct label. In other words, for
true label 2, we only care about the component of yhat corrsponding to 2. Cross-
entropy attempts to maximize the log-likelihood given to the correct labels.

```{.python .input  n=16}
def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat), axis=0, exclude=True)
```

## Optimizer

For this example we'll be using the same stochastic gradient descent (SGD)
optimizer as last time.

```{.python .input  n=17}
def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
```

## Write evaluation loop to calculate accuracy

While cross-entropy is nice, differentiable loss function, it's not the way
humans usually evaluate performance on multiple choice tasks. More commonly we
look at accuracy, the number of correct answers divided by the total number of
questions. Let's write an evaluation loop that will take a data iterator and a
network, returning the model's accuracy  averaged over the entire dataset.

```{.python .input  n=18}
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

```{.python .input  n=19}
evaluate_accuracy(test_data, net)
```

```{.json .output n=19}
[
 {
  "data": {
   "text/plain": "0.079518311"
  },
  "execution_count": 19,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Execute training loop

```{.python .input  n=21}
epochs = 10
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
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
            
#   if i % 100 == 0:
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))       
```

```{.json .output n=21}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 1.33395984883, Train_acc 0.75683, Test_acc 0.763137\nEpoch 1. Loss: 0.956869472658, Train_acc 0.813166, Test_acc 0.817874\nEpoch 2. Loss: 0.807611667182, Train_acc 0.835788, Test_acc 0.841959\nEpoch 3. Loss: 0.720267472443, Train_acc 0.848881, Test_acc 0.855693\nEpoch 4. Loss: 0.66092317514, Train_acc 0.857992, Test_acc 0.863356\nEpoch 5. Loss: 0.617147813763, Train_acc 0.864073, Test_acc 0.869924\nEpoch 6. Loss: 0.583158488127, Train_acc 0.869703, Test_acc 0.874104\nEpoch 7. Loss: 0.555837432406, Train_acc 0.873817, Test_acc 0.876791\nEpoch 8. Loss: 0.533299543665, Train_acc 0.877249, Test_acc 0.880076\nEpoch 9. Loss: 0.514308319997, Train_acc 0.880264, Test_acc 0.881867\n"
 }
]
```

## Conclusion

Jeepers. We can get nearly 90% accuracy at this task just by training a linear
model for a few seconds! You might reasonably conclude that this problem is too
easy to be taken seriously by experts.

But until recently, many papers (Google Scholar says 13,800) were published
using results obtained on this data. Even this year, I reviewed a paper whose
primary achievment was an (imagined) improvement in performance. While MNIST can
be a nice toy dataset for testing new ideas, we don't recommend writing papers
with it.

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
