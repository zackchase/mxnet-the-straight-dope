# Convolutional Neural Networks in ``gluon``

Now let's see how succinctly we can express a convolutional neural network using ``gluon``. You might be relieved to find out that this too requires hardly any more code than a logistic regression. 

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)
```

## Set the context

```{.python .input  n=2}
ctx = mx.gpu()
```

## Grab the MNIST dataset

```{.python .input  n=3}
batch_size = 64
num_inputs = 784
num_outputs = 10
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
```

## Define a convolutional neural network

Again, a few lines here is all we need in order to change the model. Let's add a couple of convolutional layers using ``gluon.nn``.

```{.python .input  n=4}
num_fc = 512
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))            
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    # The Flatten layer collapses all axis, except the first one, into one axis.
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_fc, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))
```

## Parameter initialization


```{.python .input  n=5}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Softmax cross-entropy Loss

```{.python .input  n=6}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

```{.python .input  n=7}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
```

## Write evaluation loop to calculate accuracy

```{.python .input  n=8}
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
```

## Training Loop

```{.python .input  n=9}
epochs = 10
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
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
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))    
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.0803655722426, Train_acc 0.981083333333, Test_acc 0.9817\nEpoch 1. Loss: 0.0519956610579, Train_acc 0.986983333333, Test_acc 0.9871\nEpoch 2. Loss: 0.0352727754972, Train_acc 0.989416666667, Test_acc 0.9873\nEpoch 3. Loss: 0.0281875100931, Train_acc 0.991133333333, Test_acc 0.9854\nEpoch 4. Loss: 0.0269846401864, Train_acc 0.99565, Test_acc 0.99\nEpoch 5. Loss: 0.018723519692, Train_acc 0.99515, Test_acc 0.9912\nEpoch 6. Loss: 0.0192750721679, Train_acc 0.99715, Test_acc 0.9912\nEpoch 7. Loss: 0.0132102874016, Train_acc 0.996283333333, Test_acc 0.9903\nEpoch 8. Loss: 0.010889969811, Train_acc 0.998383333333, Test_acc 0.9911\nEpoch 9. Loss: 0.012427249999, Train_acc 0.998366666667, Test_acc 0.9911\n"
 }
]
```

## Conclusion

You might notice that by using ``gluon``, we get code that runs much faster whether on CPU or GPU. That's largely because ``gluon`` can call down to highly optimized layers that have been written in C++. 

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
