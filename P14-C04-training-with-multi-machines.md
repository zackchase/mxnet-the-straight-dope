# Training with multi-machines

On previous two tutorials we saw that using multiple GPUs within a machine can
accelerate the training. The speedup, however, is limited by the number of GPUs
installed in that machine, which is typically at most 16 nowadays. This amount
of speedup is still not good enough for large scale applications, such as
training a state-of-the-art CNNs on millions of images.

This tutorial we discuss the key ideas how to turn a single machine training
program into distributed training. A typical distributed system is illustrated
in the following figure, where multiple machines are connected by network
switches.

![](img/multi-machines.svg)

Note that the preview way that using `copyto` to copy data from one GPU to
another does not work for GPUs sitting on difference machine. We need a better
abstraction here.

## Key-value store

MXNet provides a key-value store to synchronize data among devices. The
following codes initialize a `ndarray` associated with the key "weight" on a
key-value store.

```{.python .input  n=1}
from mxnet import kv, nd
store = kv.create('local')
shape = (2, 3)
x = nd.random_uniform(shape=shape)
store.init('weight', x) 
print('=== init "weight" ==={}'.format(x))
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== init \"weight\" ===\n[[ 0.54881352  0.59284461  0.71518934]\n [ 0.84426576  0.60276335  0.85794562]]\n<NDArray 2x3 @cpu(0)>\n"
 }
]
```

After initialization, we can pull the value to multiple devices.

```{.python .input  n=2}
from mxnet import gpu
ctx = [gpu(0), gpu(1)]
y = [nd.zeros(shape, ctx=c) for c in ctx]
store.pull('weight', out=y)
print('=== pull "weight" to {} ===\n{}'.format(ctx, y))
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== pull \"weight\" to [gpu(0), gpu(1)] ===\n[\n[[ 0.54881352  0.59284461  0.71518934]\n [ 0.84426576  0.60276335  0.85794562]]\n<NDArray 2x3 @gpu(0)>, \n[[ 0.54881352  0.59284461  0.71518934]\n [ 0.84426576  0.60276335  0.85794562]]\n<NDArray 2x3 @gpu(1)>]\n"
 }
]
```

We can also push new data value into the store. It will first sum the data on
the same key and then overwrite the current value.

```{.python .input  n=3}
z = [nd.ones(shape, ctx=ctx[i])+i for i in range(len(ctx))]
store.push('weight', z)
print('=== push to "weight" ===\n{}'.format(z))
store.pull('weight', out=y)
print('=== pull "weight" ===\n{}'.format(y))
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== push to \"weight\" ===\n[\n[[ 1.  1.  1.]\n [ 1.  1.  1.]]\n<NDArray 2x3 @gpu(0)>, \n[[ 2.  2.  2.]\n [ 2.  2.  2.]]\n<NDArray 2x3 @gpu(1)>]\n=== pull \"weight\" ===\n[\n[[ 3.  3.  3.]\n [ 3.  3.  3.]]\n<NDArray 2x3 @gpu(0)>, \n[[ 3.  3.  3.]\n [ 3.  3.  3.]]\n<NDArray 2x3 @gpu(1)>]\n"
 }
]
```

With `push` and `pull` we can replace the `allreduce` function defined in
[multiple-gpus-scratch](P14-C02-multiple-gpus-scratch.ipynb) by

```python
def allreduce(data, data_name, store):
    store.push(data_name, data)
    store.pull(data_name, out=data)
```

## Distributed key-value store

Not only for data synchronization within a machine, the key-value store also
supports inter-machine communication. To use it, one can create a distributed
kvstore by (Note: distributed key-value store requires `mxnet` to be compiled
with the flag `USE_DIST_KVSTORE=1`, e.g. `make USE_DIST_KVSTORE=1`.)

```python
store = kv.create('dist')
```

Now if we run the codes from the last section on two machines at the same time,
then the store will aggregate the two ndarrays pushed from each machine, after
that, the pulled results will be

```
[[ 6.  6.  6.]
 [ 6.  6.  6.]]
```

In the distributed setting, `MXNet` launches three kinds of processes (each time
running `python myprog.py` will create a process). One is *worker*, which runs
the user programs, such as the codes on the last section. The other twos are
*server*, which maintains the data pushed into the store, and *scheduler*, which
monitors the aliveness of each node.

It's up to users which machines to run these processes. But to simplify the
process placement and launching, MXNet provides a tool located at
[tools/launch.py](https://github.com/dmlc/mxnet/blob/master/tools/launch.py).

Assume there are two machines, A and B. They are ssh-able, and their IPs are
saved in a file named `hostfile`. Then we can start one worker in each machine
through:

```
$ mxnet_path/tools/launch.py -H hostfile -n 2 python myprog.py
```

It will also start a server in each machine, and the scheduler on the same
machine we are currently on.

![](img/dist_kv.svg)

## Use `kvstore` in `gluon`

As mentioned in [multiple-gpu-scratch](P14-C02-multiple-gpus-scratch.ipynb#data-
parallelism), to implement data parallelism we just need to specify
- how to split data
- how to synchronize gradients and weights

We already see from [multiple-gpu-gluon](P14-C03-multiple-gpus-gluon.ipynb#put-
all-things-together) that a `gluon` trainer can automatically aggregate the
gradients among different GPUs. What it really does is having a key-value store
with type `local` within it. Therefore, to change to multi-machine training we
only need to pass a distributed key-value store, for example,

```python
store = kv.create('dist')
trainer = gluon.Trainer(..., kvstore=store)
```

To split the data, however, we cannot directly copy the previous approach. One
commonly used solution is to split the whole dataset into *k* parts at the
beginning, then let the *i*-th worker only reads the *i*-th part of the data.

We can obtain the total number of workers by the attribute `num_workers` and the
rank of the current worker by the attribute `rank`.

```{.python .input  n=4}
print('total number of workers: %d'%(store.num_workers))
print('my rank among workers: %d'%(store.rank))
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "total number of workers: 1\nmy rank among workers: 0\n"
 }
]
```

With this information, we can manually seek to a proper position of the input
data. In addition, several data iterators provided by `MXNet` already support to
read a part of the data. For example,

```python
from mxnet.io import ImageRecordIter
data = ImageRecordIter(num_parts=store.num_workers, part_index=store.rank, ...)
```

```{.python .input}

```
