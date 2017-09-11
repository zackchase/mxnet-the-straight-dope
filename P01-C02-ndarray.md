# Manipulate data the MXNet way with ``ndarray``

It's impossible to get anything done if we can't manipulate data. So let's start
by introducing NDArrays, MXNet's primary tool for storing and transforming data.
If you've worked with NumPy before, you'll notice that NDArrays are by design
similar similar to NumPy's multi-dimensional array. However, they confer a few
key advantages. First, NDArrays support asynchronous computation on CPU, GPU,
and distributed cloud architectures. Second, they provide support for automatic
differentiation. These properties make NDArray an ideal library for machine
learning, both for researchers and engineers launching production systems.


## Getting started

In this chapter, we'll get you going with the basic functionality. Don't worry
if you don't understand any of the basic math, like element-wise operations or
normal distributions. In the next two chapters we'll take another pass at
NDArray, teaching you both the math you'll need and how to realize it in code.

To get started, let's import ``mxnet`` and (for convenience) ``mxnet.ndarray``,
the only dependencies we'll need in this tutorial. We'll also make a habit of
setting a random seed so that you always get the same results that I do.

```{.python .input  n=3}
import mxnet as mx
import mxnet.ndarray as nd
mx.random.seed(1)
```

Next, let's see how to create an NDArray, without initializing values.
Speficially we'll create a 2D array (also called a *matrix*) with 6 rows and 4
columns.

```{.python .input  n=4}
x = nd.empty(shape=(6,4))
print(x)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ -9.62835260e+20   4.56024559e-41   3.34257530e-37   0.00000000e+00]\n [ -1.13891193e+21   4.56024559e-41  -2.18271178e+20   4.56024559e-41]\n [ -7.33561680e+30   4.56010546e-41  -1.13939157e+21   4.56024559e-41]\n [ -2.57377060e+20   4.56024559e-41  -2.55593634e+20   4.56024559e-41]\n [ -2.54911339e+20   4.56024559e-41  -1.14559049e+20   4.56024559e-41]\n [ -7.30426210e+30   4.56010546e-41  -1.15180799e+21   4.56024559e-41]]\n<NDArray 6x4 @cpu(0)>\n"
 }
]
```

Often, we'll want to create arrays whose values are sampled randomly. This is
especially common when we intend to use the array as a parameter in a neural
network. In this snippet, we initialize with values drawn from a standard normal
distribution.

```{.python .input  n=5}
x = nd.random_normal(shape=(6,4))
print(x)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.67765152  0.03629481  0.10073948 -0.49024421]\n [ 0.57595438 -0.95017916 -0.3469252   0.03751944]\n [-0.22134334 -0.72984636 -1.80471897 -2.04010558]\n [-0.80642909  1.482131    1.22033095  1.04082799]\n [ 2.23235631 -0.45256865  0.20070229  0.31160426]\n [-0.54968649 -0.83673775 -0.19819015 -0.78830057]]\n<NDArray 6x4 @cpu(0)>\n"
 }
]
```

As in NumPy, the dimensions of each NDArray are accessible via the ``.shape``
attribute.

```{.python .input  n=6}
print(x.shape)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(6, 4)\n"
 }
]
```

We can also query its size, which is equal to the product of the components of
the shape. Together with the precision of the stored values, this tells us how
much memory the array occupies.

```{.python .input  n=7}
print(x.size)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "24\n"
 }
]
```

## Operations

NDarray supports a large number of standard mathematical operations.

```{.python .input  n=8}
y = nd.random_normal(shape=(6,4))
c = x + y
print(c)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-1.06343007  0.16293958  1.47182953 -0.66866344]\n [ 0.33805454 -0.18538713 -0.19824563 -0.18024123]\n [-0.7198565   1.49594152 -2.65287685 -2.72843218]\n [-0.72831756  2.32368684  0.83791792  0.42498678]\n [ 2.17901564 -0.43295699 -0.10932122  0.0673877 ]\n [-2.28340673 -0.97094506 -1.26003861 -1.30634451]]\n<NDArray 6x4 @cpu(0)>\n"
 }
]
```

## In-place operations

In the previous example, we allocated new memory for the sum ``x+y`` and
assigned a reference to the variable ``c``. To make better use of memory, we
often prefer to perform operations in place, reusing already allocated memory.

In MXNet, we can specify where to write the results of operations by assigning
them with slice notation, e.g., ``result[:] = ...``.

```{.python .input  n=9}
result = nd.zeros(shape=(6,4))
result[:] = x+y
print(result)
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-1.06343007  0.16293958  1.47182953 -0.66866344]\n [ 0.33805454 -0.18538713 -0.19824563 -0.18024123]\n [-0.7198565   1.49594152 -2.65287685 -2.72843218]\n [-0.72831756  2.32368684  0.83791792  0.42498678]\n [ 2.17901564 -0.43295699 -0.10932122  0.0673877 ]\n [-2.28340673 -0.97094506 -1.26003861 -1.30634451]]\n<NDArray 6x4 @cpu(0)>\n"
 }
]
```

If we're not planning to re-use ``x``, then we can assign the result to ``x``
itself.

```{.python .input  n=10}
x[:] = x + y
```

In MXNet, the ``+=`` operator performs an in place addition. Note that ``x +=
y`` does not allocate new memory while ``x = x + y`` does.

```{.python .input  n=20}
##################
#  y += x overwrites the array referenced by y 
#  (the same array is also referenced by x)
##################
x = nd.ones(shape=(1,1))
y = x
y += x
print(y == x)

##################
#  y = y + x allocates a new array and assigns to y
#  (now x and y point to different arrays)
##################
x = nd.ones(shape=(1,1))
y = x
y = y + x
print(y == x)
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 1.]]\n<NDArray 1x1 @cpu(0)>\n\n[[ 0.]]\n<NDArray 1x1 @cpu(0)>\n"
 }
]
```

But be careful! This is **NOT** the same as ``x = x + y``. If we don't use slice
notation then we allocate new memory and assign a reference to the new data to
the variable ``x``.

## Slicing

MXNet NDArrays support slicing in all the ridiculous ways you might imagine
accessing your data. Here's an example of reading the second and third rows from
``x``.

```{.python .input  n=9}
x[2:4]
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "\n[[-0.7198565   1.49594152 -2.65287685 -2.72843218]\n [-0.72831756  2.32368684  0.83791792  0.42498678]]\n<NDArray 2x4 @cpu(0)>"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Now let's try writing to a specific element.

```{.python .input  n=10}
x[3,2] = 9.0
print(x[3])
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[-0.72831756  2.32368684  9.          0.42498678]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

## Weird multi-dimensional slicing

We can even write to arbitrary ranges along each of the axes.

```{.python .input  n=11}
x[2:4,1:3] = 5.0
print(x)
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-1.06343007  0.16293958  1.47182953 -0.66866344]\n [ 0.33805454 -0.18538713 -0.19824563 -0.18024123]\n [-0.7198565   5.          5.         -2.72843218]\n [-0.72831756  5.          5.          0.42498678]\n [ 2.17901564 -0.43295699 -0.10932122  0.0673877 ]\n [-2.28340673 -0.97094506 -1.26003861 -1.30634451]]\n<NDArray 6x4 @cpu(0)>\n"
 }
]
```

## Converting from MXNet NDArray to NumPy

Converting MXNet NDArrays to and from NumPy is easy. Note that, unlike in
PyTorch, the converted arrays do not share memory.

```{.python .input  n=12}
a = nd.ones(shape=(5))
print(a)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 1.  1.  1.  1.  1.]\n<NDArray 5 @cpu(0)>\n"
 }
]
```

```{.python .input  n=13}
b = a.asnumpy()
print(b)
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[ 1.  1.  1.  1.  1.]\n"
 }
]
```

```{.python .input  n=14}
b[0] = 2
print(b)
print(a)
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[ 2.  1.  1.  1.  1.]\n\n[ 1.  1.  1.  1.  1.]\n<NDArray 5 @cpu(0)>\n"
 }
]
```

## Converting from NumPy Array to MXNet NDArray

Constructing an MXNet NDarray from a NumPy Array is straightforward.

```{.python .input  n=15}
c = nd.array(b)
print(c)
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 2.  1.  1.  1.  1.]\n<NDArray 5 @cpu(0)>\n"
 }
]
```

## Managing context

In MXNet, every array has a context. One context could be the CPU. Other
contexts might be various GPUs. Things can get even hairier when we deploy jobs
across multiple servers. By assigning arrays to contexts intelligently, we can
minimize the time spent transferring data between devices. For example, when
training neural networks on a server with a GPU, we typically prefer for the
model's parameters to live on the GPU. To start, let's try initializing an array
on the CPU.


```{.python .input  n=16}
d = nd.array(b, mx.cpu())
```

Given an NDArray on a given context, we can copy it to another context by using
the ``copyto()`` method.

```{.python .input  n=17}
e = d.copyto(mx.gpu(0))
print(e)
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 2.  1.  1.  1.  1.]\n<NDArray 5 @gpu(0)>\n"
 }
]
```

## Watch out!

Imagine that your variable ``d`` already lives on your second GPU
(``mx.gpu(1)``). What happens if we call ``d.copyto(mx.gpu(1))``? It will make a
copy and allocate new memory, even though that variable already lives on the
desired device!

Often, we only want to make a copy if the variable *currently* lives in the
wrong context. In these cases, we can call ``as_in_context()``. If the variable
is already on ``mx.gpu(1)`` then this is a no-op.

```{.python .input  n=18}
f = d.as_in_context(mx.cpu(0))
print(f)
```

```{.json .output n=18}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 2.  1.  1.  1.  1.]\n<NDArray 5 @cpu(0)>\n"
 }
]
```

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

## Broadcasting

You might wonder, what happens if you add a vector ``y`` to a matrix ``X``?
These operations, where we compose a low dimensional array ``y`` with a high-
dimensional array ``X`` invoke a functionality called broadcasting. Here, the
low-dimensional array is duplicated along any axis with dimension ``1`` to match
the shape of the high dimesnional array. Consider the following example.

```{.python .input  n=19}
X = nd.ones(shape=(4,4))
y = nd.arange(4)
print(y)
print(y.shape)
print(X+y)
```

```{.json .output n=19}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 0.  1.  2.  3.]\n<NDArray 4 @cpu(0)>\n(4,)\n\n[[ 1.  2.  3.  4.]\n [ 1.  2.  3.  4.]\n [ 1.  2.  3.  4.]\n [ 1.  2.  3.  4.]]\n<NDArray 4x4 @cpu(0)>\n"
 }
]
```

While ``y`` is initially of shape (4,), MXNet infers its shape to be (1,4), and
then broadcasts along the rows to form a (4,4) matrix). You might wonder, why
did MXNet choose to interpret ``y`` as a (1,4) matrix and not (4,1). That's
because broadcasting prefers to duplicate along the left most axis. We can alter
this behavior by explicitly giving ``y`` a 2D shape.

```{.python .input  n=20}
y = nd.arange(4).reshape((4,1))
print(y)
print(y.shape)
print(X+y)
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.]\n [ 1.]\n [ 2.]\n [ 3.]]\n<NDArray 4x1 @cpu(0)>\n(4, 1)\n\n[[ 1.  1.  1.  1.]\n [ 2.  2.  2.  2.]\n [ 3.  3.  3.  3.]\n [ 4.  4.  4.  4.]]\n<NDArray 4x4 @cpu(0)>\n"
 }
]
```

For whinges or inquiries, [open an issue on
GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
