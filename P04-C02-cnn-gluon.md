# Convolutional Neural Networks in ``gluon``

Now let's see how succinctly we can express a convolutional neural network using
``gluon``. You might be relieved to find out that this too requires hardly any
more code than a logistic regression.

```{.python .input  n=3}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)
```

## Set the context

```{.python .input  n=4}
ctx = mx.gpu()
```

## Grab the MNIST dataset

```{.python .input  n=5}
mnist = mx.test_utils.get_mnist()
batch_size = 64
train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size, shuffle=True)
test_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size, shuffle=True)
```

## Define a convolutional neural network

Again, a few lines here is all we need in order to change the model. Let's add a
couple convolutional layers using ``gluon.nn``.

```{.python .input  n=6}
#########################
#   Can do it with sequential once nn.Faltten() gets merged 
#########################

# net = gluon.nn.Sequential()
# with net.name_scope():
#     net.add(gluon.Conv2D(channels=20, kernel_size=3, activation=‘relu')
#     net.add(gluon.Conv2D(channels=50, kernel_size=5, activation=‘relu')
#     net.add(gluon.nn.Flatten())
#     net.add(gluon.nn.Dense(128, activation="relu"))
#     net.add(gluon.nn.Dense(10))


class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = gluon.nn.Conv2D(20, kernel_size=(5,5))
            self.pool1 = gluon.nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.conv2 = gluon.nn.Conv2D(50, kernel_size=(5,5))
            self.pool2 = gluon.nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.fc1 = gluon.nn.Dense(500)
            self.fc2 = gluon.nn.Dense(10)

    def forward(self, x):
        x = self.pool1(nd.tanh(self.conv1(x)))
        x = self.pool2(nd.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = nd.tanh(self.fc1(x))
        x = nd.tanh(self.fc2(x))
        return x
    
net = Net()
```

## Parameter initialization


```{.python .input  n=7}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Softmax cross-entropy Loss

```{.python .input  n=8}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

```{.python .input  n=9}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
```

## Write evaluation loop to calculate accuracy

```{.python .input  n=10}
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()
```

## Training Loop

```{.python .input  n=11}
epochs = 10
moving_loss = 0.

for e in range(epochs):
    train_data.reset()
    for i, batch in enumerate(train_data):
        data = batch.data[0].as_in_context(ctx)
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
            moving_loss = nd.mean(cross_entropy).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(cross_entropy).asscalar()
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))    
    
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.878388477348, Train_acc 0.970666, Test_acc 0.973925\nEpoch 1. Loss: 0.845597321857, Train_acc 0.980927, Test_acc 0.981986\nEpoch 2. Loss: 0.833567160561, Train_acc 0.985524, Test_acc 0.985171\nEpoch 3. Loss: 0.826708851309, Train_acc 0.98814, Test_acc 0.986764\nEpoch 4. Loss: 0.821894711464, Train_acc 0.989589, Test_acc 0.988157\nEpoch 5. Loss: 0.818307372079, Train_acc 0.990972, Test_acc 0.988555\nEpoch 6. Loss: 0.815555502798, Train_acc 0.992104, Test_acc 0.989351\nEpoch 7. Loss: 0.813360093674, Train_acc 0.992521, Test_acc 0.98955\nEpoch 8. Loss: 0.811664162928, Train_acc 0.992954, Test_acc 0.989351\nEpoch 9. Loss: 0.810315940825, Train_acc 0.993387, Test_acc 0.989252\n"
 }
]
```

## Conclusion

You might notice that by using ``gluon``, we get code that runs much faster
whether on CPU or GPU. That's largely because ``gluon`` can call down to highly
optimized layers that have been written in C++.

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
