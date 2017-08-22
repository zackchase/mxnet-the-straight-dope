# Training with multi-GPUs from scratch

Nowadays, it is common that a single machine is equipped with more than one GPU. The following figure shows that 4 GPUs are connected to the CPU through a PCIe switch. 

![](img/multi-gpu.svg)

If NVIDIA driver is installed, we can check how many GPUs are available by running the command `nvidia-smi`. Note that running this tutorial requires at least 2 GPUs. 

```{.python .input  n=1}
!nvidia-smi
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sun Jul 23 09:01:41 2017       \n+-----------------------------------------------------------------------------+\n| NVIDIA-SMI 375.66                 Driver Version: 375.66                    |\n|-------------------------------+----------------------+----------------------+\n| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |\n|===============================+======================+======================|\n|   0  GRID K520           Off  | 0000:00:03.0     Off |                  N/A |\n| N/A   42C    P0    42W / 125W |      0MiB /  4036MiB |      0%      Default |\n+-------------------------------+----------------------+----------------------+\n|   1  GRID K520           Off  | 0000:00:04.0     Off |                  N/A |\n| N/A   34C    P0    41W / 125W |      0MiB /  4036MiB |      0%      Default |\n+-------------------------------+----------------------+----------------------+\n|   2  GRID K520           Off  | 0000:00:05.0     Off |                  N/A |\n| N/A   32C    P0    42W / 125W |      0MiB /  4036MiB |      0%      Default |\n+-------------------------------+----------------------+----------------------+\n|   3  GRID K520           Off  | 0000:00:06.0     Off |                  N/A |\n| N/A   28C    P0    37W / 125W |      0MiB /  4036MiB |      0%      Default |\n+-------------------------------+----------------------+----------------------+\n                                                                               \n+-----------------------------------------------------------------------------+\n| Processes:                                                       GPU Memory |\n|  GPU       PID  Type  Process name                               Usage      |\n|=============================================================================|\n|  No running processes found                                                 |\n+-----------------------------------------------------------------------------+\n"
 }
]
```

Using all of the GPUs together often significantly speeds up the training. As compared to CPU where all the CPU cores are used by default, most layers of a neural network can only run on a single GPU. Therefore, additional workers are required to partition the workload into multiple GPUs. 

## Data Parallelism

Data parallelism is a widely used approach to partition workloads. Assume there are *k* GPUs, it splits the examples in a data batch into *k* parts, such that each GPU will compute the gradient on a part of the batch. Gradients are then summed across all GPUs before updating the weights.

The following pseudo codes shows how to train one data batch on *k* GPUs. 


```
def train_batch(data, k):
    split data into k parts
    for i = 1, ..., k:  # run in parallel
        compute grad_i w.r.t. weight_i using data_i on the i-th GPU
    grad = grad_1 + ... + grad_k
    for i = 1, ..., k:  # run in parallel
        copy grad to i-th GPU
        update weight_i by using grad
```

Next we will present how to implement this algorithm from scratch.


## Automatic Parallelization

We first demonstrate how to run workloads in parallel. Writing parallel codes in Python in non-trivial, but fortunately, MXNet is able to automatically parallelize the workloads. Two technologies are used to achieve this goal.

Firstly, workloads, such as `nd.dot` are pushed into the backend engine for *lazy evaluation*. That is, Python merely pushes the workload `nd.dot` and returns immediately without waiting for the computation to be finished. We keep pushing until the results need to be copied out from MXNet, such as `print(x)` or are converted into numpy by `x.asnumpy()`. At that time, the Python thread is blocked until the results are ready.

```{.python .input  n=2}
from mxnet import nd
from time import time

start = time()
x = nd.random_uniform(shape=(2000,2000))
y = nd.dot(x, x)
print('=== workloads are pushed into the backend engine ===\n%f sec' % (time() - start))
z = y.asnumpy()
print('=== workloads are finished ===\n%f sec' % (time() - start))
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== workloads are pushed into the backend engine ===\n0.000778 sec\n=== workloads are finished ===\n0.254895 sec\n"
 }
]
```

Secondly, the backend engine will analyse the dependencies of the pushed workloads. If two workloads are independent of each other, then the engine may run them in parallel.

For example, if we issue three operators:

```
a = nd.random_uniform(...)
b = nd.random_uniform(...)
c = a + b
```

Then `a` and `b` may run in parallel, while `c` needs to wait until both `a` and `b` are ready. 

The following codes show that the engine effectively parallelizes the `dot` operations on two GPUs

```{.python .input  n=3}
from mxnet import gpu

def run(x):
    """push 10 matrix-matrix multiplications"""
    return [nd.dot(x,x) for i in range(10)]

def wait(x):
    """explicitly wait until all results are ready"""
    for y in x:
        y.wait_to_read()

x0 = nd.random_uniform(shape=(4000, 4000), ctx=gpu(0))
x1 = x0.copyto(gpu(1))

print('=== Run on GPU 0 and 1 in sequential ===')
start = time()
wait(run(x0))
wait(run(x1))
print('time: %f sec' %(time() - start))

print('=== Run on GPU 0 and 1 in parallel ===')
start = time()
y0 = run(x0)
y1 = run(x1)
wait(y0)
wait(y1)
print('time: %f sec' %(time() - start))
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== Run on GPU 0 and 1 in sequential ===\ntime: 2.305928 sec\n=== Run on GPU 0 and 1 in parallel ===\ntime: 1.080638 sec\n"
 }
]
```

```{.python .input  n=4}
from mxnet import cpu

def copy(x, ctx):
    """copy data to a device"""
    return [y.copyto(ctx) for y in x]

print('=== Run on GPU 0 and then copy results to CPU in sequential ===')
start = time()
y0 = run(x0)
wait(y0)
z0 = copy(y0, cpu())
wait(z0)
print(time() - start)

print('=== Run and copy in parallel ===')
start = time()
y0 = run(x0)
z0 = copy(y0, cpu())
wait(z0)
print(time() - start)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== Run on GPU 0 and then copy results to CPU in sequential ===\n1.36029291153\n=== Run and copy in parallel ===\n1.10839509964\n"
 }
]
```

## Define model and updater

We will use the convolutional neural networks and plain SGD introduced in [cnn-scratch]() as an example workload.

```{.python .input  n=5}
from mxnet import gluon
# initialize parameters
scale = .01
W1 = nd.random_normal(shape=(20,1,3,3))*scale
b1 = nd.zeros(shape=20)
W2 = nd.random_normal(shape=(50,20,5,5))*scale
b2 = nd.zeros(shape=50)
W3 = nd.random_normal(shape=(800,128))*scale
b3 = nd.zeros(shape=128)
W4 = nd.random_normal(shape=(128,10))*scale
b4 = nd.zeros(shape=10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# network and loss
def lenet(X, params):
    # first conv
    h1_conv = nd.Convolution(data=X, weight=params[0], bias=params[1], kernel=(3,3), num_filter=20)
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    # second conv
    h2_conv = nd.Convolution(data=h1, weight=params[2], bias=params[3], kernel=(5,5), num_filter=50)
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    h2 = nd.flatten(h2)
    # first fullc
    h3_linear = nd.dot(h2, params[4]) + params[5]
    h3 = nd.relu(h3_linear)
    # second fullc    
    yhat = nd.dot(h3, params[6]) + params[7]    
    return yhat

loss = gluon.loss.SoftmaxCrossEntropyLoss()

# plain SGD
def SGD(params, lr):
    for p in params:
        p[:] = p - lr * p.grad
```

## Utility functions to synchronize data across GPUs

The following function copies the parameters into a particular GPU and initializes the gradients.

```{.python .input  n=6}
def get_params(params, ctx):
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params

new_params = get_params(params, gpu(0))
print('=== copy b1 to GPU(0) ===\nweight = {}\ngrad = {}'.format(
    new_params[1], new_params[1].grad))
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== copy b1 to GPU(0) ===\nweight = \n[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n  0.  0.]\n<NDArray 20 @gpu(0)>\ngrad = \n[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n  0.  0.]\n<NDArray 20 @gpu(0)>\n"
 }
]
```

Given a list of data that span over multiple GPUs, we define a function to sum the data and then broadcast the results to each GPU. 

```{.python .input  n=7}
def allreduce(data):
    # sum on data[0].context, and then broadcast
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)
    for i in range(1, len(data)):
        data[0].copyto(data[i])        

data = [nd.ones((1,2), ctx=gpu(i))*(i+1) for i in range(2)]
print("=== before allreduce ===\n {}".format(data))
allreduce(data)
print("\n=== after allreduce ===\n {}".format(data))
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== before allreduce ===\n [\n[[ 1.  1.]]\n<NDArray 1x2 @gpu(0)>, \n[[ 2.  2.]]\n<NDArray 1x2 @gpu(1)>]\n\n=== after allreduce ===\n [\n[[ 3.  3.]]\n<NDArray 1x2 @gpu(0)>, \n[[ 3.  3.]]\n<NDArray 1x2 @gpu(1)>]\n"
 }
]
```

Given a data batch, we define a function that splits this batch and copies each part into the corresponding GPU.

```{.python .input  n=8}
def split_and_load(data, ctx):
    n, k = data.shape[0], len(ctx)
    assert (n//k)*k == n, '# examples is not divided by # devices'
    idx = list(range(0, n+1, n//k))
    return [data[idx[i]:idx[i+1]].as_in_context(ctx[i]) for i in range(k)]

batch = nd.arange(16).reshape((4,4))
print('=== original data ==={}'.format(batch))
ctx = [gpu(0), gpu(1)]
splitted = split_and_load(batch, ctx)
print('\n=== splitted into {} ==={}\n{}'.format(ctx, splitted[0], splitted[1]))
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== original data ===\n[[  0.   1.   2.   3.]\n [  4.   5.   6.   7.]\n [  8.   9.  10.  11.]\n [ 12.  13.  14.  15.]]\n<NDArray 4x4 @cpu(0)>\n\n=== splitted into [gpu(0), gpu(1)] ===\n[[ 0.  1.  2.  3.]\n [ 4.  5.  6.  7.]]\n<NDArray 2x4 @gpu(0)>\n\n[[  8.   9.  10.  11.]\n [ 12.  13.  14.  15.]]\n<NDArray 2x4 @gpu(1)>\n"
 }
]
```

## Train and inference one data batch

Now we are ready to implement how to train one data batch with data parallelism.

```{.python .input  n=9}
def train_batch(batch, params, ctx, lr):
    # split the data batch and load them on GPUs
    data = split_and_load(batch.data[0], ctx)
    label = split_and_load(batch.label[0], ctx)
    # run forward on each GPU
    with gluon.autograd.record():
        losses = [loss(lenet(X, W), Y) 
                  for X, Y, W in zip(data, label, params)]
    # run backward on each gpu
    for l in losses:
        l.backward()
    # aggregate gradient over GPUs
    for i in range(len(params[0])):                
        allreduce([params[c][i].grad for c in range(len(ctx))])
    # update parameters with SGD on each GPU
    for p in params:
        SGD(p, lr/batch.data[0].shape[0])
```

For inference, we simply let it run on the first GPU. We leave a data parallelism implementation as an exercise.

```{.python .input  n=10}
def valid_batch(batch, params, ctx):
    data = batch.data[0].as_in_context(ctx[0])
    pred = nd.argmax(lenet(data, params[0]), axis=1)
    return nd.sum(pred == batch.label[0].as_in_context(ctx[0])).asscalar()
```

## Put all things together

Define the program that trains and validates the model on MNIST. 

```{.python .input  n=11}
from mxnet.test_utils import get_mnist
from mxnet.io import NDArrayIter

def run(num_gpus, batch_size, lr):    
    # the list of GPUs will be used
    ctx = [gpu(i) for i in range(num_gpus)]
    print('Running on {}'.format(ctx))
    
    # data iterator
    mnist = get_mnist()
    train_data = NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size)
    valid_data = NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)
    print('Batch size is {}'.format(batch_size))
    
    # copy parameters to all GPUs
    dev_params = [get_params(params, c) for c in ctx]
    for epoch in range(5):
        # train
        start = time()
        train_data.reset()
        for batch in train_data:
            train_batch(batch, dev_params, ctx, lr)
        nd.waitall()  # wait all computations are finished to benchmark the time
        print('Epoch %d, training time = %.1f sec'%(epoch, time()-start))
        
        # validating
        valid_data.reset()
        correct, num = 0.0, 0.0
        for batch in valid_data:
            correct += valid_batch(batch, dev_params, ctx)
            num += batch.data[0].shape[0]                
        print('         validation accuracy = %.4f'%(correct/num))

```

First run on a single GPU with batch size 64.

```{.python .input  n=12}
run(1, 64, 0.3)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Running on [gpu(0)]\nBatch size is 64\nEpoch 0, training time = 5.3 sec\n         validation accuracy = 0.9551\nEpoch 1, training time = 4.9 sec\n         validation accuracy = 0.9769\nEpoch 2, training time = 4.9 sec\n         validation accuracy = 0.9823\nEpoch 3, training time = 4.9 sec\n         validation accuracy = 0.9845\nEpoch 4, training time = 5.0 sec\n         validation accuracy = 0.9810\n"
 }
]
```

Running on multiple GPUs, we often want to increase the batch size so that each GPU still gets a large enough batch size for good computation performance. A larger batch size sometimes slow down the convergence, we often want to increases the learning rate as well.  

```{.python .input  n=13}
run(2, 128, 0.6)
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Running on [gpu(0), gpu(1)]\nBatch size is 128\nEpoch 0, training time = 3.2 sec\n         validation accuracy = 0.9255\nEpoch 1, training time = 3.0 sec\n         validation accuracy = 0.9656\nEpoch 2, training time = 3.2 sec\n         validation accuracy = 0.9762\nEpoch 3, training time = 3.2 sec\n         validation accuracy = 0.9790\nEpoch 4, training time = 3.1 sec\n         validation accuracy = 0.9831\n"
 }
]
```

## Conclusion

We have shown how to develop a data parallelism training program from scratch. Thanks to the auto-parallelism, we only need to write serial codes while the engine is able to parallelize them on multiple GPUs.

```{.python .input}

```
