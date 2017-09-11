# Linear regression with ``gluon``

Now that we've implemented a whole neural network from scratch, using nothing
but ``mx.ndarray`` and ``mxnet.autograd``, let's see how we can make the same
model while doing a lot less work.

Again, let's import some packages, this time adding ``mxnet.gluon`` to the list
of dependencies.

```{.python .input  n=11}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
```

## Set the context

And let's also set a context where we'll do most of the computation.

```{.python .input  n=12}
ctx = mx.cpu()
```

## Build the dataset

Again we'll look at the problem of linear regression and stick with synthetic
data.

```{.python .input  n=13}
X = nd.random_normal(shape=(10000,2))
y = 2* X[:,0] - 3.4 * X[:,1] + 4.2 + .01 * nd.random_normal(shape=(10000,))
```

## Load the data iterator

We'll stick with the ``NDArrayIter`` for handling out data batching

```{.python .input  n=14}
batch_size = 4
train_data = mx.io.NDArrayIter(X, y, batch_size, shuffle=True)
```

## Define the model

Before we had to individual allocate our parameters and then compose them as a
model. While it's good to know how to do things from scratch, with ``gluon``, we
can usually just compose a network from predefined standard layers.

```{.python .input  n=15}
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
```

## Initialize parameters

Before we can do anything with this model we'll have to initialize the weights.
*MXNet* provides a variety of common initializers in ``mxnet.init``. Note that
we pass the initializer a context. That's how we tell ``gluon`` model where
should to store our parameters. Once we start training deep nets, we'll
generally want to keep parameters on one or more GPUs.

```{.python .input  n=16}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Define loss

Instead of writing our own loss function wer'e just going to call down to
``gluon.loss.L2Loss``

```{.python .input  n=17}
loss = gluon.loss.L2Loss()
```

## Optimizer

Instead of writing gradient descent from scratch every time, we can instantiate
a ``gluon.Trainer``, passing it a dictionary of parameters.

```{.python .input  n=18}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
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

```{.python .input  n=19}
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
            mse = loss(output, label)
        mse.backward()
        trainer.step(data.shape[0])
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        if i == 0:
            moving_loss = nd.mean(mse).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(mse).asscalar()
            
        if i % 500 == 0:
            print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i, moving_loss))    
```

```{.json .output n=19}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0, batch 0. Moving avg of loss: 26.3618\nEpoch 0, batch 500. Moving avg of loss: 0.178251259028\nEpoch 0, batch 1000. Moving avg of loss: 0.00122218602843\nEpoch 0, batch 1500. Moving avg of loss: 5.98630845125e-05\nEpoch 0, batch 2000. Moving avg of loss: 5.06800330684e-05\nEpoch 1, batch 0. Moving avg of loss: 3.4354e-05\nEpoch 1, batch 500. Moving avg of loss: 5.61056995295e-05\nEpoch 1, batch 1000. Moving avg of loss: 5.1357795241e-05\nEpoch 1, batch 1500. Moving avg of loss: 5.21701774608e-05\nEpoch 1, batch 2000. Moving avg of loss: 5.06294869531e-05\n"
 }
]
```

## Conclusion

As you can see, even for a simple eample like linear regression, ``gluon`` can
help you to write quick, clean, clode. Next, we'll repeat this exercise for
multilayer perceptrons, extending these lessons to deep neural networks and
(comparatively) real datasets.

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
