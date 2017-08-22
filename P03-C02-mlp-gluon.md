# Multilayer perceptrons in ``gluon``

Using gluon, we only need two additional lines of code to transform our logisitc regression model into a multilayer perceptron.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from mxnet import gluon
```

We'll also want to set the compute context for our modeling. Feel free to go ahead and change this to mx.gpu(0) if you're running on an appropriately endowed machine.

```{.python .input  n=2}
ctx = mx.cpu()
```

## The MNIST dataset

```{.python .input  n=3}
mnist = mx.test_utils.get_mnist()
batch_size = 64
num_inputs = 784
num_outputs = 10
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
```

## Define the model

*Here's the only real difference. We add two lines!*

```{.python .input  n=4}
num_hidden = 256
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))
```

## Parameter initialization


```{.python .input  n=5}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Softmax cross-entropy loss

```{.python .input  n=6}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

```{.python .input  n=7}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
```

## Evaluation metric

```{.python .input  n=8}
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
```

## Training loop

```{.python .input  n=9}
epochs = 10
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
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
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.208460539446, Train_acc 0.948683333333, Test_acc 0.9482\nEpoch 1. Loss: 0.137320037022, Train_acc 0.958966666667, Test_acc 0.9551\nEpoch 2. Loss: 0.0958231976158, Train_acc 0.956716666667, Test_acc 0.9492\nEpoch 3. Loss: 0.0725868264617, Train_acc 0.98395, Test_acc 0.9754\nEpoch 4. Loss: 0.0646171670057, Train_acc 0.9836, Test_acc 0.9735\nEpoch 5. Loss: 0.0469602448996, Train_acc 0.987683333333, Test_acc 0.9766\nEpoch 6. Loss: 0.0403166358583, Train_acc 0.99195, Test_acc 0.9783\nEpoch 7. Loss: 0.034311452392, Train_acc 0.991866666667, Test_acc 0.977\nEpoch 8. Loss: 0.0319601120719, Train_acc 0.994733333333, Test_acc 0.9783\nEpoch 9. Loss: 0.0243036117522, Train_acc 0.991466666667, Test_acc 0.977\n"
 }
]
```

## Conclusion

We showed the much simpler way to define a multilayer perceptrons in ``gluon``. Now let's take a look at how to build convolutional neural networks.

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
