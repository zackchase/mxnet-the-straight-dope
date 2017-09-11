# Automatic differentiation with ``autograd``


In machine learning, we *train* models to get better and better as a function of
experience. Usually, *getting better* means minimizing a *loss function*, i.e. a
score that answers "how *bad* is our model?" With neural networks, we choose
loss functions to be differentiable with respect to our parameters.

Put simply, this means that for each of the model's parameters, we can determine
how much *increasing* or *decreasing* it might affect the loss. While the
calculations are straightforward, for complex models, working it out by hand can
be a pain.

_MXNet_'s autograd package expedites this work by automatically calculating
derivatives. And while most other libraries require that we compile a symbolic
graph to take automatic derivatives, ``mxnet.autograd``, like PyTorch, allows
you to take derivatives while writing  ordinary imperative code.

Every time you make pass through your model, ``autograd`` builds a graph on the
fly, through which it can immediately backpropagate gradients.

Let's go through it step by step. For this tutorial, we'll only need to import
``mxnet.ndarray``, and ``mxnet.autograd``.

```{.python .input  n=1}
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.autograd as ag
mx.random.seed(1)
```

## Attaching gradients

As a toy example, Let's say that we are interested in differentiating a function
``f = 2 * (x ** 2)`` with respect to parameter x. We can start by assigning an
initial value of ``x``.

```{.python .input  n=2}
x = nd.array([[1, 2], [3, 4]])
```

Once we compute the gradient of ``f`` with respect to ``x``, we'll need a place
to store it. In _MXNet_, we can tell an NDArray that we plan to store a gradient
by invoking its ``attach_grad()`` method.

```{.python .input  n=3}
x.attach_grad()
```

Now we're going to define out function ``f`` and *MXNet* will generate a
computation graph on the fly. It's as if *MXNet* turned on a recording device
and captured the exact path by which each variable was generated.

Note that building the computation graph requires a nontrivial amount of
computation. So we only *MXNet* to build the graph when explicitly told to do
so. We can instruct *MXNet* to start recording by placing code inside a ``with
autograd.record():`` block.

```{.python .input  n=4}
with ag.record():
  y = x * 2
  z = y * x
```

Let's backprop by calling ``z.backward()``. When ``z`` has more than one entry,
``z.backward() is equivalent to mx.nd.sum(z).backward().



```{.python .input  n=5}
z.backward()
```

Now, let's see if this is the expected output. Remember that ``y = x * 2``, and
``z = x * y``, so ``z`` should be equal to  ``2 * x * x``. After, doing backprop
with ``z.backward()``, we expect to get back gradient dz/dx as follows: dy/dx =
``2``, dz/dx = ``4 * x``. So, if everything went according to plan, ``x.grad``
should consist of an NDArray with the values ``[[4, 8],[12, 16]]``.

```{.python .input  n=6}
print(x.grad)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[  4.   8.]\n [ 12.  16.]]\n<NDArray 2x2 @cpu(0)>\n"
 }
]
```

## Head gradients and the chain rule

*Warning: This part is tricky, but not necessary to understanding subsequent
sections.*

Sometimes when we call the backward method on an NDArray, e.g. ``y.backward()``,
where ``y`` is a function of ``x`` we are just interested in the derivative of
``y`` with respect to ``x``. At other times, we may be interested in the
gradient of ``z`` with respect to ``x``, where ``z`` is a function of ``y``.
Recall that by the chain rule dz/dx can be expressed in terms of dz/dy and
dy/dx. So, when ``y`` is part of a larger function ``z``, and we want ``x.grad``
to store dz/dx, we can pass in the *head gradient* dz/dy as an input to
``backward()``. The default argument is ``nd.ones_like(y)``.

```{.python .input  n=7}
with ag.record():
  y = x * 2
  z = y * x
    
head_gradient = nd.array([[10,1.],[.1,.01]])
z.backward(head_gradient)
print(x.grad)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 40.           8.        ]\n [  1.20000005   0.16      ]]\n<NDArray 2x2 @cpu(0)>\n"
 }
]
```

Now that we know the basics, we can do some wild things with autograd, including
building diferentiable functions using Pythonic control flow.

```{.python .input  n=8}
a = nd.random_normal(shape=3)
a.attach_grad()

with ag.record():
    b = a * 2
    while (nd.norm(b) < 1000).asscalar():
        b = b * 2

    if (mx.nd.sum(b) > 0).asscalar():
        c = b
    else :
        c = 100 * b
```

```{.python .input  n=9}
head_gradient = nd.array([0.01,1.0,.1])
c.backward(head_gradient)
```

```{.python .input  n=10}
print(a.grad)
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[   2048.  204800.   20480.]\n<NDArray 3 @cpu(0)>\n"
 }
]
```

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
