# Multilayer perceptrons in ``gluon``

Using gluon, we only need two additional lines of code to transform our logisitc
regression model into a multilayer perceptron.

```{.python .input  n=55}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
```

We'll also want to set the compute context for our modeling. Feel free to go
ahead and change this to mx.gpu(0) if you're running on an appropriately endowed
machine.

```{.python .input  n=56}
ctx = mx.cpu()
```

## The MNIST dataset

```{.python .input  n=57}
mnist = mx.test_utils.get_mnist()
batch_size = 64
train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size, shuffle=True)
test_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size, shuffle=True)
```

## Define the model

*Here's the only real difference. We add two lines!*

```{.python .input  n=64}
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(128, activation="relu"))
    net.add(gluon.nn.Dense(64, activation="relu"))
    net.add(gluon.nn.Dense(10))
```

## Parameter initialization


```{.python .input  n=65}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Softmax cross-entropy loss

```{.python .input  n=66}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

```{.python .input  n=61}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
```

## Evaluation metric

```{.python .input  n=62}
metric = mx.metric.Accuracy()

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    
    data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        with autograd.record():
            data = batch.data[0].as_in_context(ctx).reshape((-1,784))
            label = batch.label[0].as_in_context(ctx)
            label_one_hot = nd.one_hot(label, 10)
            output = net(data)
        
        metric.update([label], [output])
    return metric.get()[1]
```

## Training loop

```{.python .input  n=63}
epochs = 10
moving_loss = 0.

for e in range(epochs):
    train_data.reset()
    for i, batch in enumerate(train_data):
        data = batch.data[0].as_in_context(ctx).reshape((-1,784))
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
            moving_loss = np.mean(cross_entropy.asnumpy()[0])
        else:
            moving_loss = .99 * moving_loss + .01 * np.mean(cross_entropy.asnumpy()[0])
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))    
    
```

```{.json .output n=63}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.23931532364, Train_acc 0.946860730594, Test_acc 0.945262738854\nEpoch 1. Loss: 0.14545513, Train_acc 0.955507990868, Test_acc 0.948669628594\nEpoch 2. Loss: 0.0970559549169, Train_acc 0.961448820396, Test_acc 0.956427087772\nEpoch 3. Loss: 0.0713375280292, Train_acc 0.965760559361, Test_acc 0.961963429692\nEpoch 4. Loss: 0.0562312310473, Train_acc 0.969003995434, Test_acc 0.966091304827\nEpoch 5. Loss: 0.0471335310744, Train_acc 0.971729927702, Test_acc 0.969227183949\nEpoch 6. Loss: 0.0375632641638, Train_acc 0.973980756686, Test_acc 0.971890330013\nEpoch 7. Loss: 0.0306285313321, Train_acc 0.975852597032, Test_acc 0.974077521734\nEpoch 8. Loss: 0.0244164425957, Train_acc 0.977409944191, Test_acc 0.975897513177\nEpoch 9. Loss: 0.0171164385061, Train_acc 0.978745719178, Test_acc 0.977425526868\n"
 }
]
```

## Conclusion

Now let's take a look at how to build convolutional neural networks.

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
