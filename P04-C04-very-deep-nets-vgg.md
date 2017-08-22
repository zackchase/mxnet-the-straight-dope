# Very deep networks with repeating elements

As we already noticed in AlexNet, the number of layers in networks keeps on increasing. This means that it becomes extremely tedious to write code that piles on one layer after the other manually. Fortunately, programming languages have a wonderful fix for this: subroutines and loops. This way we can express networks as *code*. Just like we would use a for loop to count from 1 to 10, we'll use code to combine layers. The first network that had this structure was VGG. 

## VGG

We begin with the usual import ritual

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)
```

```{.python .input  n=2}
ctx = mx.gpu()
```

## Load up a dataset


```{.python .input  n=3}
def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype(np.float32)
    return data, label
```

```{.python .input  n=4}
batch_size = 32
train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=False, transform=transformer),
    batch_size=batch_size, shuffle=False, last_batch='discard')
```

```{.python .input  n=5}
for d, l in train_data:
    break
```

```{.python .input  n=6}
print(d.shape, l.shape)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(32, 3, 224, 224) (32,)\n"
 }
]
```

```{.python .input  n=7}
d.dtype
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "numpy.float32"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## The VGG architecture

A key aspect of VGG was to use many convolutional blocks with relatively narrow kernels, followed by a max-pooling step and to repeat this block multiple times. What is pretty neat about the code below is that we use functions to *return* network blocks. These are then combined to larger networks (e.g. in `vgg_stack`) and this allows us to construct `vgg_net` from components. What is particularly useful here is that we can use it to reparametrize the architecture simply by changing a few lines rather than adding and removing many lines of network definitions. 

```{.python .input  n=8}
def vgg_block(convs, channels):
    out = gluon.nn.HybridSequential(prefix='')
    for i in range(convs):
        out.add(gluon.nn.Conv2D(channels=channels, kernel_size=3, activation='relu'))
    out.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))    
    return out

def vgg_stack(architecture):
    out = gluon.nn.HybridSequential(prefix='')
    for (convs, channels) in architecture:
        out.add(vgg_block(convs, channels))
    return out

num_classes = 10
architecture = ((2,64), (2,128), (3,256), (3, 512), (3, 512))
vgg_net = gluon.nn.HybridSequential()
with vgg_net.name_scope():
    vgg_net.add(vgg_stack(architecture))
    # Flatten and apply fullly connected layers
    vgg_net.add(gluon.nn.Flatten())
    vgg_net.add(gluon.nn.Dense(4096, activation="relu"))
    vgg_net.add(gluon.nn.Dense(4096, activation="relu"))
    vgg_net.add(gluon.nn.Dense(num_classes))
# speed up execution with hybridization
vgg_net.hybridize()
```

## Initialize parameters

```{.python .input  n=9}
vgg_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Optimizer

```{.python .input  n=10}
trainer = gluon.Trainer(vgg_net.collect_params(), 'sgd', {'learning_rate': .01})
```

## Softmax cross-entropy loss

```{.python .input  n=11}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Evaluation loop

```{.python .input  n=12}
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
```

## Training loop

```{.python .input  n=13}
###########################
#  Only one epoch so tests can run quickly, increase this variable to actually run
###########################
epochs = 1
smoothing_constant = .01

for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = vgg_net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        
        if i > 0 and i % 200 == 0:
            print('Batch %d. Loss: %f' % (i, moving_loss))
            
    test_accuracy = evaluate_accuracy(test_data, vgg_net)
    train_accuracy = evaluate_accuracy(train_data, vgg_net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))    
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Batch 200. Loss: 2.284135\nBatch 400. Loss: 2.241947\nBatch 600. Loss: 2.163716\nBatch 800. Loss: 2.142409\nBatch 1000. Loss: 2.189017\nBatch 1200. Loss: 2.090356\nBatch 1400. Loss: 2.002341\nEpoch 0. Loss: 1.9552065177, Train_acc 0.23095390525, Test_acc 0.2265625\n"
 }
]
```

```{.python .input}

```
