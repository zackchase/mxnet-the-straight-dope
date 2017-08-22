# Faster, and portable through hybridizing

The tutorials we saw so far adopt the *imperative*, or define-by-run, programming paradigm. This is how we write Python programs. Another commonly used programming paradigm by deep learning frameworks is the *symbolic*, or define-then-run, programming. It consists of three steps:

- define the workloads, such as creating the neural network
- compile the program into a front-end language, e.g. Python, independent format
- feed with data to run

This compilation step may optimize the program to be more efficient to run, and also the resulting language-independent format make the program portable to various front-end languages. 

`gluon` provides a *hybrid* mechanism to seamless combine both declarative programming and imperative programming. Users can freely switch between them to enjoy the advantages of both paradigms. 

## HybridSequential

We already learned how to use `Sequential` to stack the layers. Now, we have `HybridSequential` that constructs a hybrid network. Its usage is similar to `Sequential`:

```{.python .input  n=5}
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd

def get_net():
    # construct a MLP
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(2))
    # initialize the parameters
    net.collect_params().initialize()
    return net

# forward
x = nd.random_normal(shape=(1, 512))
net = get_net()
print('=== net(x) ==={}'.format(net(x)))
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== net(x) ===\n[[-0.11171372  0.13755465]]\n<NDArray 1x2 @cpu(0)>\n"
 }
]
```

You can call the `hybridize` method on a `Block` to activate compile and optimize it. Only `HybridBlock`s, e.g. `HybridSequential`, can be compiled. But you can still call `hybridize` on normal `Block` and its `HybridBlock` children will be compiled instead. We will talk more about `HybridBlock` later.

```{.python .input  n=6}
net.hybridize()
print('=== net(x) ==={}'.format(net(x)))
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== net(x) ===\n[[-0.11171372  0.13755465]]\n<NDArray 1x2 @cpu(0)>\n"
 }
]
```

## Performance

We compare the performance between before hybridizing and after hybridizing by forwarding 1000 times.

```{.python .input  n=8}
from time import time
def bench(net, x):
    mx.nd.waitall()
    start = time()
    for i in range(1000):
        y = net(x)
    mx.nd.waitall()
    return time() - start
        
net = get_net()
print('Before hybridizing: %.4f sec'%(bench(net, x)))
net.hybridize()
print('After hybridizing: %.4f sec'%(bench(net, x)))
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Before hybridizing: 0.3096 sec\nAfter hybridizing: 0.1652 sec\n"
 }
]
```

As can been seen, there is a significant speedup after hybridizing.

## Get the symbolic program

Previously, we feed `net` with `NDArray` data `x`, and then `net(x)` returned the forward results. Now if we feed it with a `Symbol` placeholder, then the corresponding symbolic program will be returned. 

```{.python .input  n=9}
from mxnet import sym
x = sym.var('data')
print('=== input data holder ===')
print(x)

y = net(x)
print('\n=== the symbolic program of net===')
print(y)

y_json = y.tojson()
print('\n=== the according json definition===')
print(y_json)
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== input data holder ===\n<Symbol data>\n\n=== the symbolic program of net===\n<Symbol hybridsequential4_dense2_fwd>\n\n=== the according json definition===\n{\n  \"nodes\": [\n    {\n      \"op\": \"null\", \n      \"name\": \"data\", \n      \"inputs\": []\n    }, \n    {\n      \"op\": \"null\", \n      \"name\": \"hybridsequential4_dense0_weight\", \n      \"attr\": {\n        \"__dtype__\": \"0\", \n        \"__lr_mult__\": \"1.0\", \n        \"__shape__\": \"(256, 0)\", \n        \"__wd_mult__\": \"1.0\"\n      }, \n      \"inputs\": []\n    }, \n    {\n      \"op\": \"null\", \n      \"name\": \"hybridsequential4_dense0_bias\", \n      \"attr\": {\n        \"__dtype__\": \"0\", \n        \"__init__\": \"zeros\", \n        \"__lr_mult__\": \"1.0\", \n        \"__shape__\": \"(256,)\", \n        \"__wd_mult__\": \"1.0\"\n      }, \n      \"inputs\": []\n    }, \n    {\n      \"op\": \"FullyConnected\", \n      \"name\": \"hybridsequential4_dense0_fwd\", \n      \"attr\": {\"num_hidden\": \"256\"}, \n      \"inputs\": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]\n    }, \n    {\n      \"op\": \"Activation\", \n      \"name\": \"hybridsequential4_dense0_relu_fwd\", \n      \"attr\": {\"act_type\": \"relu\"}, \n      \"inputs\": [[3, 0, 0]]\n    }, \n    {\n      \"op\": \"null\", \n      \"name\": \"hybridsequential4_dense1_weight\", \n      \"attr\": {\n        \"__dtype__\": \"0\", \n        \"__lr_mult__\": \"1.0\", \n        \"__shape__\": \"(128, 0)\", \n        \"__wd_mult__\": \"1.0\"\n      }, \n      \"inputs\": []\n    }, \n    {\n      \"op\": \"null\", \n      \"name\": \"hybridsequential4_dense1_bias\", \n      \"attr\": {\n        \"__dtype__\": \"0\", \n        \"__init__\": \"zeros\", \n        \"__lr_mult__\": \"1.0\", \n        \"__shape__\": \"(128,)\", \n        \"__wd_mult__\": \"1.0\"\n      }, \n      \"inputs\": []\n    }, \n    {\n      \"op\": \"FullyConnected\", \n      \"name\": \"hybridsequential4_dense1_fwd\", \n      \"attr\": {\"num_hidden\": \"128\"}, \n      \"inputs\": [[4, 0, 0], [5, 0, 0], [6, 0, 0]]\n    }, \n    {\n      \"op\": \"Activation\", \n      \"name\": \"hybridsequential4_dense1_relu_fwd\", \n      \"attr\": {\"act_type\": \"relu\"}, \n      \"inputs\": [[7, 0, 0]]\n    }, \n    {\n      \"op\": \"null\", \n      \"name\": \"hybridsequential4_dense2_weight\", \n      \"attr\": {\n        \"__dtype__\": \"0\", \n        \"__lr_mult__\": \"1.0\", \n        \"__shape__\": \"(2, 0)\", \n        \"__wd_mult__\": \"1.0\"\n      }, \n      \"inputs\": []\n    }, \n    {\n      \"op\": \"null\", \n      \"name\": \"hybridsequential4_dense2_bias\", \n      \"attr\": {\n        \"__dtype__\": \"0\", \n        \"__init__\": \"zeros\", \n        \"__lr_mult__\": \"1.0\", \n        \"__shape__\": \"(2,)\", \n        \"__wd_mult__\": \"1.0\"\n      }, \n      \"inputs\": []\n    }, \n    {\n      \"op\": \"FullyConnected\", \n      \"name\": \"hybridsequential4_dense2_fwd\", \n      \"attr\": {\"num_hidden\": \"2\"}, \n      \"inputs\": [[8, 0, 0], [9, 0, 0], [10, 0, 0]]\n    }\n  ], \n  \"arg_nodes\": [0, 1, 2, 5, 6, 9, 10], \n  \"node_row_ptr\": [\n    0, \n    1, \n    2, \n    3, \n    4, \n    5, \n    6, \n    7, \n    8, \n    9, \n    10, \n    11, \n    12\n  ], \n  \"heads\": [[11, 0, 0]], \n  \"attrs\": {\"mxnet_version\": [\"int\", 1100]}\n}\n"
 }
]
```

Now we can save both the program and parameters onto disk, so that it can be loaded later not only in Python, but in all other supported languages, such as C++, R, and Scala, as well.

```{.python .input  n=5}
y.save('model.json')
net.save_params('model.params')
```

## HybridBlock

Now let's dive deeper into how `hybridize` works. Remember that another way to construct a network is to define a subclass of `gluon.Block`, by which we can flexibly write the forward function. 

Unsurprisingly, there is a hybridized version `HybridBlock`. We implement the previous MLP as: 

```{.python .input  n=10}
from mxnet import gluon

class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(256)
            self.fc2 = nn.Dense(128)
            self.fc3 = nn.Dense(2)

    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym
        print('type(x): {}, F: {}'.format(
                type(x).__name__, F.__name__))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

Now we feed data into the network, we can see that `hybrid_forward` is called twice.

```{.python .input  n=11}
net = Net()
net.collect_params().initialize()
x = nd.random_normal(shape=(1, 512))
print('=== 1st forward ===')
y = net(x)
print('=== 2nd forward ===')
y = net(x)
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== 1st forward ===\ntype(x): NDArray, F: mxnet.ndarray\n=== 2nd forward ===\ntype(x): NDArray, F: mxnet.ndarray\n"
 }
]
```

Now run it again after hybridizing. 

```{.python .input  n=12}
net.hybridize()
print('=== 1st forward ===')
y = net(x)
print('=== 2nd forward ===')
y = net(x)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== 1st forward ===\ntype(x): Symbol, F: mxnet.symbol\n=== 2nd forward ===\n"
 }
]
```

It differs from the previous execution in two aspects:

1. the input data type now is `Symbol` even when we fed a `NDArray` into `net`, because `gluon` implicitly constructed a symbolic data placeholder.
2. `hybrid_forward` is called once at the first time we run `net(x)`. It is because `gluon` will construct the symbolic program on the first forward, and then keep it for reuse later.

One main reason that the network is faster after hybridizing is because we don't need to repeatedly invoke the Python forward function, while keeping all computations within the highly efficient C++ backend engine.

But the potential drawback is the loss of flexibility to write the forward function. In other ways, inserting `print` for debugging or control logic such as `if` and `for` into the forward function is not possible now.

## Conclusion

Through `HybridSequental` and `HybridBlock`, we can convert an imperative program into a symbolic program by calling `hybridize`. 
