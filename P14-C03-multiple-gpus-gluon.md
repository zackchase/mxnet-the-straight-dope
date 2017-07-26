#  Training with multiple GPUs with `gluon`

Now we are going to implement the data parallelism algorithm introduced in
[multi gpu from scratch](./P14-C03-multiple-gpus-gluon.ipynb) with `gluon`.

First we define the example network and loss function.

```{.python .input  n=1}
from mxnet import gluon, gpu
net = gluon.nn.Sequential(prefix='cnn_')
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=3, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(128, activation="relu"))
    net.add(gluon.nn.Dense(10))
    
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Initialize on multiple devices

`gluon` supports initialize the network parameters over multiple devices. The
parameters on one device is identical to the ones on another device.

```{.python .input  n=2}
ctx = [gpu(0), gpu(1)]
net.collect_params().initialize(ctx=ctx)
```

Given input data, the parameters on the according device are used to compute the
results.

```{.python .input  n=3}
from mxnet.test_utils import get_mnist
mnist = get_mnist()
batch = mnist['train_data'][0:4, :]
data = gluon.utils.split_and_load(batch, ctx)
print(net(data[0]))
print(net(data[1]))
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.01017658  0.03012515  0.02999702  0.01175333 -0.01746453  0.00707828\n   0.02404996  0.00616632 -0.02094562  0.0136827 ]\n [-0.01249129  0.0305641   0.02823936 -0.00159418 -0.00722831  0.00538148\n   0.01476716  0.0225275  -0.02458289  0.0246105 ]]\n<NDArray 2x10 @gpu(0)>\n\n[[-0.00349744  0.01896121  0.02959755  0.00261514  0.00015916 -0.00355723\n   0.0040103   0.03075583 -0.00761715  0.00599077]\n [-0.00557119  0.02766508  0.02406837 -0.0007478  -0.00511122  0.00538528\n   0.00292899  0.01488838 -0.00191687  0.01074106]]\n<NDArray 2x10 @gpu(1)>\n"
 }
]
```

We can access the parameters on each device. (Note that, the weights may be
initialized at the beginning of the first forward, while not in `initialize`
because the data shapes may be not available at that time).

```{.python .input  n=4}
weight = net.collect_params()['cnn_conv2d0_weight']

for c in ctx:
    print('=== channel 0 of the first conv2d on {} ==={}'.format(
        c, weight.data(ctx=c)[0]))
    
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== channel 0 of the first conv2d on gpu(0) ===\n[[[ 0.0068339   0.01299825  0.0301265 ]\n  [ 0.04819721  0.01438687  0.05011239]\n  [ 0.00628365  0.04861524 -0.01068833]]]\n<NDArray 1x3x3 @gpu(0)>\n=== channel 0 of the first conv2d on gpu(1) ===\n[[[ 0.0068339   0.01299825  0.0301265 ]\n  [ 0.04819721  0.01438687  0.05011239]\n  [ 0.00628365  0.04861524 -0.01068833]]]\n<NDArray 1x3x3 @gpu(1)>\n"
 }
]
```

Similar we can access the gradients on each GPUs. Because the input data are
different, the gradients on each GPU vary.

```{.python .input  n=5}
def forward_backward(net, data, label):
    with gluon.autograd.record():
        losses = [loss(net(X), Y) for X, Y in zip(data, label)]
    for l in losses:
        l.backward()
        
label = gluon.utils.split_and_load(mnist['train_label'][0:4], ctx)
forward_backward(net, data, label)
for c in ctx:
    print('=== grad of channel 0 of the first conv2d on {} ==={}'.format(
        c, weight.grad(ctx=c)[0]))
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== grad of channel 0 of the first conv2d on gpu(0) ===\n[[[-0.00481181  0.02549154  0.05066926]\n  [ 0.01503928  0.04740802  0.04111018]\n  [ 0.04527877  0.06305876  0.04087965]]]\n<NDArray 1x3x3 @gpu(0)>\n=== grad of channel 0 of the first conv2d on gpu(1) ===\n[[[-0.01102538 -0.02251887 -0.02211753]\n  [-0.01587106 -0.03848278 -0.03960424]\n  [-0.03371563 -0.06092874 -0.064744  ]]]\n<NDArray 1x3x3 @gpu(1)>\n"
 }
]
```

## Put all things together

Now we can implement the remaining functions. Most of them are the same as the
previous tutorial, one notable difference is that a `gluon` trainer recognizes
multi-devices, it will automatically aggregate the gradients and synchronize the
parameters.

```{.python .input  n=6}
from mxnet import nd
from mxnet.io import NDArrayIter
from time import time

def train_batch(batch, ctx, net, trainer):
    # split the data batch and load them on GPUs
    data = gluon.utils.split_and_load(batch.data[0], ctx)
    label = gluon.utils.split_and_load(batch.label[0], ctx)
    # compute gradient
    forward_backward(net, data, label)
    # update parameters
    trainer.step(batch.data[0].shape[0])
    
def valid_batch(batch, ctx, net):
    data = batch.data[0].as_in_context(ctx[0])
    pred = nd.argmax(net(data), axis=1)
    return nd.sum(pred == batch.label[0].as_in_context(ctx[0])).asscalar()    

def run(num_gpus, batch_size, lr):    
    # the list of GPUs will be used
    ctx = [gpu(i) for i in range(num_gpus)]
    print('Running on {}'.format(ctx))
    
    # data iterator
    mnist = get_mnist()
    train_data = NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size)
    valid_data = NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)
    print('Batch size is {}'.format(batch_size))
    
    net.collect_params().initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    for epoch in range(5):
        # train
        start = time()
        train_data.reset()
        for batch in train_data:
            train_batch(batch, ctx, net, trainer)
        nd.waitall()  # wait all computations are finished to benchmark the time
        print('Epoch %d, training time = %.1f sec'%(epoch, time()-start))
        
        # validating
        valid_data.reset()
        correct, num = 0.0, 0.0
        for batch in valid_data:
            correct += valid_batch(batch, ctx, net)
            num += batch.data[0].shape[0]                
        print('         validation accuracy = %.4f'%(correct/num))
        
run(1, 64, .3)        
run(2, 128, .6)            
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Running on [gpu(0)]\nBatch size is 64\nEpoch 0, training time = 4.9 sec\n         validation accuracy = 0.9764\nEpoch 1, training time = 4.7 sec\n         validation accuracy = 0.9839\nEpoch 2, training time = 4.6 sec\n         validation accuracy = 0.9836\nEpoch 3, training time = 4.7 sec\n         validation accuracy = 0.9861\nEpoch 4, training time = 4.7 sec\n         validation accuracy = 0.9862\nRunning on [gpu(0), gpu(1)]\nBatch size is 128\nEpoch 0, training time = 2.8 sec\n         validation accuracy = 0.8951\nEpoch 1, training time = 2.8 sec\n         validation accuracy = 0.9687\nEpoch 2, training time = 2.8 sec\n         validation accuracy = 0.9759\nEpoch 3, training time = 2.8 sec\n         validation accuracy = 0.9785\nEpoch 4, training time = 2.9 sec\n         validation accuracy = 0.9813\n"
 }
]
```

## Conclusion

Both parameters and trainers in `gluon` supports multi-devices, moving from one
device to multi-devices is straightforward.
