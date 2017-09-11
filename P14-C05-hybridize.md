# Faster, and portable through hybridizing

The tutorials we saw so far adopt the *imperative*, or define-by-run,
programming paradigm. This is the way how we write Python programs. Another
commonly used programming paradigm by deep learning frameworks is the
*symbolic*, or define-then-run, programming. It consists of three steps

- define the workloads, such as creating the neural network
- compile the program into a front-end language, e.g. Python, independent format
- feed with data to run

This compilation step may optimize the program to be more efficient to run, and
also the resulted language independent format make the program portable to
various front-end languages.

`gluon` provides a *hybrid* mechanism to seamless combine both declarative
programming and imperative programming. Users can freely switch between them to
enjoy the advantages of both paradigms.

## HybridSequential

We already learned how to use `Sequential` to stack the layers, there is
`HybridSequential` that construct a hybrid network. Its usage is similar to
`Sequential`:

```{.python .input  n=1}
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

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== net(x) ===\n[[ 0.09178199 -0.01948221]]\n<NDArray 1x2 @cpu(0)>\n"
 }
]
```

The network constructed by `HybridSequential` can be called `hybridize` to hint
`gluon` to compile it through the symbolic way.

```{.python .input  n=2}
net.hybridize()
print('=== net(x) ==={}'.format(net(x)))
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== net(x) ===\n[[ 0.09178199 -0.01948221]]\n<NDArray 1x2 @cpu(0)>\n"
 }
]
```

## Performance

We compare the performance between before hybridizing and after hybridizing by
forwarding 1000 times.

```{.python .input  n=3}
from time import time
def bench(net, x):
    start = time()
    for i in range(1000):
        y = net(x)
    y.wait_to_read()
    return time() - start
        
net = get_net()
print('Before hybridizing: %.4f sec'%(bench(net, x)))
net.hybridize()
print('After hybridizing: %.4f sec'%(bench(net, x)))
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Before hybridizing: 0.8890 sec\nAfter hybridizing: 0.4335 sec\n"
 }
]
```

As can been seen, there is a significant speedup after hybridizing.

## Get the symbolic program

Previous we feed `net` with data `x`, then `net(x)` returns the forward results.
Now if we feed it with a symbolic data placeholder, then the according symbolic
program will be returned.

```{.python .input  n=4}
from mxnet import sym
x = sym.var('data')
print('=== input data holder ===')
print(x)

y = net(x)
print('\n=== the symbolic program of net===')
print(y)

y_json = y.tojson()
print('\n=== the according json definition===')
print(y_json[0:200])
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== input data holder ===\n<Symbol data>\n\n=== the symbolic program of net===\n<Symbol fullyconnected14>\n\n=== the according json definition===\n{\n  \"nodes\": [\n    {\n      \"op\": \"null\", \n      \"name\": \"data\", \n      \"inputs\": []\n    }, \n    {\n      \"op\": \"null\", \n      \"name\": \"hybridsequential1_dense0_weight\", \n      \"attr\": {\n        \"__dtyp\n"
 }
]
```

Now we can save both the program and parameters into disk, so that it can be
loaded later not only on Python, but all other supported languages, such as C++,
R, and Scalar, as well.

```{.python .input  n=5}
y.save('model.json')
net.collect_params().save('model.params')
```

## HybridBlock

Now let dive deeper into how `hybridize` works. Remember that another way to
construct a network is define a subclass of `gluon.Block`, by it we can flexibly
write the forward function.

Not surprise there a hybridized version `HybridBlock`. We implement the previous
MLP:

```{.python .input  n=6}
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

Now we feed data into the network, we can see that `hybrid_forward` is called
twice.

```{.python .input  n=7}
net = Net()
net.collect_params().initialize()
x = nd.random_normal(shape=(1, 512))
print('=== 1st forward ===')
y = net(x)
print('=== 2nd forward ===')
y = net(x)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== 1st forward ===\ntype(x): NDArray, F: mxnet.ndarray\n=== 2nd forward ===\ntype(x): NDArray, F: mxnet.ndarray\n"
 }
]
```

Now run it again after hybridizing.

```{.python .input  n=8}
net.hybridize()
print('=== 1st forward ===')
y = net(x)
print('=== 2nd forward ===')
y = net(x)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "=== 1st forward ===\ntype(x): Symbol, F: mxnet.symbol\n=== 2nd forward ===\n"
 }
]
```

It differs to the previous execution in two aspects:

1. the input data type now is `Symbol` even that we feed a `NDArray` into
`net`. `gluon` implicitly constructed a symbolic data placeholder.
2. `hybrid_forward` is called once at the first time we run `net(x)`. It is
because `gluon` will construct the symbolic program on the first forward, and
then keep reuse it later.

One main reason that the network is faster after hybridizing is because we don't
need to repeatedly evoke the Python forward function, while keep all
computations within the highly efficient C++ backend engine.

But the potential drawback is the losing of flexibility to write the forward
function. In other ways, inserting `print` for debugging or control logics such
as `if` and `for` into the forward function is useless now.

## Conclusion

Through `HybridSequental` and `HybridBlock`, we can convert an imperative
program into symbolic by calling `hybridize`.

```{.python .input}

```
