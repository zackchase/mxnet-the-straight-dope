#  Long short-term memory (LSTM) RNNs

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
ctx = mx.gpu(0)
```

## Dataset: "The Time Machine" 

```{.python .input  n=2}
with open("data/nlp/timemachine.txt") as f:
    time_machine = f.read()
time_machine = time_machine[:-38083]
```

## Numerical representations of characters

```{.python .input  n=3}
character_list = list(set(time_machine))
vocab_size = len(character_list)
character_dict = {}
for e, char in enumerate(character_list):
    character_dict[char] = e
time_numerical = [character_dict[char] for char in time_machine]
```

## One-hot representations

```{.python .input  n=4}
def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result
```

```{.python .input  n=5}
def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result
```

## Preparing the data for training

```{.python .input  n=7}
batch_size = 32
seq_length = 64
# -1 here so we have enough characters for labels later
num_samples = (len(time_numerical) - 1) // seq_length
dataset = one_hots(time_numerical[:seq_length*num_samples]).reshape((num_samples, seq_length, vocab_size))
num_batches = len(dataset) // batch_size
train_data = dataset[:num_batches*batch_size].reshape((num_batches, batch_size, seq_length, vocab_size))
# swap batch_size and seq_length axis to make later access easier
train_data = nd.swapaxes(train_data, 1, 2)
```

## Preparing our labels

```{.python .input  n=8}
labels = one_hots(time_numerical[1:seq_length*num_samples+1])
train_label = labels.reshape((num_batches, batch_size, seq_length, vocab_size))
train_label = nd.swapaxes(train_label, 1, 2)
```

## Long short-term memory (LSTM) RNNs

An LSTM block has mechanisms to enable "memorizing" information for an extended number of time steps. We use the LSTM block with the following transformations that map inputs to outputs across blocks at consecutive layers and consecutive time steps:

$\newcommand{\xb}{\mathbf{x}} \newcommand{\RR}{\mathbb{R}}$

$$g_t = \text{tanh}(X_t W_{xg} + h_{t-1} W_{hg} + b_g),$$

$$i_t = \sigma(X_t W_{xi} + h_{t-1} W_{hi} + b_i),$$

$$f_t = \sigma(X_t W_{xf} + h_{t-1} W_{hf} + b_f),$$

$$o_t = \sigma(X_t W_{xo} + h_{t-1} W_{ho} + b_o),$$

$$c_t = f \odot c_{t-1} + g_t \odot i_t,$$

$$h_t = \text{tanh}(c_t) \odot o_t,$$

where $\odot$ is an element-wise multiplication operator, and
for all $\xb = [x_1, x_2, \ldots, x_k]^\top \in \RR^k$ the two activation functions:
* $\sigma(\xb) = \big[1/[1+\exp(-x_1)], \ldots, 1/[1+\exp(-x_k)]\big]^\top$
* $\text{tanh}(\xb) = \big[[1-\exp(-2x_1)]/[1+\exp(-2x_1)],  \ldots, [1-\exp(-2x_k)]/[1+\exp(-2x_k)]\big]^\top$.



In the transformations above, the memory cell $c_t$ stores the "long-term" memory in the vector form.
In other words, the information accumulatively captured and  encoded  until time step $t$ is stored in $c_t$ and is only passed along the same layer over different time steps.

Given the inputs $c_t$ and $h_t$, the input gate $i_t$ and forget gate $f_t$ will help the memory cell to decide how to overwrite or keep the memory information. The output gate $o_t$ further lets the LSTM block decide how to retrieve the memory information to generate the current state $h_t$ that is passed to both the next layer of the current time step and the next time step of the current layer. Such decisions are made using the hidden-layer parameters $W$ and $b$ with different subscripts: these parameters will be inferred during the training phase by ``gluon``.


## Allocate parameters

```{.python .input  n=9}
num_inputs = 77
num_hidden = 256
num_outputs = 77

########################
#  Weights connecting the inputs to the hidden layer
########################
Wxg = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxi = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxf = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxo = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01

########################
#  Recurrent weights connecting the hidden layer across time steps
########################
Whg = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Whi = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Whf = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Who = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01

########################
#  Bias vector for hidden layer
########################
bg = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bi = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bf = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bo = nd.random_normal(shape=num_hidden, ctx=ctx) * .01


########################
# Weights to the output nodes
########################
Why = nd.random_normal(shape=(num_hidden,num_inputs), ctx=ctx) * .01
by = nd.random_normal(shape=num_inputs, ctx=ctx) * .01
```

## Attach the gradients

```{.python .input  n=10}
params = [Wxg, Wxi, Wxf, Wxo, Whg, Whi, Whf, Who, bg, bi, bf, bo]
params += [Why, by]

for param in params:
    param.attach_grad()
for param in params:
    param.attach_grad()
```

## Softmax Activation

```{.python .input  n=11}
def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition =nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition
```

## Define the model

```{.python .input  n=12}
def lstm_rnn(inputs, h, c, temperature=1.0):
    outputs = []
    for X in inputs:
        g = nd.tanh(nd.dot(X, Wxg) + nd.dot(h, Whg) + bg)
        i = nd.sigmoid(nd.dot(X, Wxi) + nd.dot(h, Whi) + bi)
        f = nd.sigmoid(nd.dot(X, Wxf) + nd.dot(h, Whf) + bf)
        o = nd.sigmoid(nd.dot(X, Wxo) + nd.dot(h, Who) + bo)
        #######################
        #
        #######################
        c = f * c + g * i
        h = nd.tanh(c * o)
        #######################
        #
        #######################
        yhat_linear = nd.dot(h, Why) + by
        yhat = softmax(yhat_linear, temperature=temperature) 
        outputs.append(yhat)
    return (outputs, h, c)
```

## Cross-entropy loss function

```{.python .input  n=13}
def cross_entropy(yhat, y):
    return - nd.mean(nd.sum(y * nd.log(yhat), axis=0, exclude=True))
```

## Averaging the loss over the sequence

```{.python .input  n=14}
def average_ce_loss(outputs, labels):
    assert(len(outputs) == len(labels))
    total_loss = 0.
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)
```

## Optimizer

```{.python .input  n=15}
def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
```

## Generating text by sampling

```{.python .input  n=16}
def sample(prefix, num_chars, temperature=1.0):
    #####################################
    # Initialize the string that we'll return to the supplied prefix
    #####################################
    string = prefix

    #####################################
    # Prepare the prefix as a sequence of one-hots for ingestion by RNN
    #####################################
    prefix_numerical = [character_dict[char] for char in prefix]
    input = one_hots(prefix_numerical)
    
    #####################################
    # Set the initial state of the hidden representation ($h_0$) to the zero vector
    #####################################    
    h = nd.zeros(shape=(1, num_hidden), ctx=ctx)
    c = nd.zeros(shape=(1, num_hidden), ctx=ctx)

    #####################################
    # For num_chars iterations,
    #     1) feed in the current input
    #     2) sample next character from from output distribution
    #     3) add sampled character to the decoded string
    #     4) prepare the sampled character as a one_hot (to be the next input)
    #####################################    
    for i in range(num_chars):
        outputs, h, c = lstm_rnn(input, h, c, temperature=temperature)
        choice = np.random.choice(77, p=outputs[-1][0].asnumpy())
        string += character_list[choice]
        input = one_hots([choice])
    return string
```

```{.python .input}
epochs = 2000
moving_loss = 0.

learning_rate = 2.0

# state = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
for e in range(epochs):
    ############################
    # Attenuate the learning rate by a factor of 2 every 100 epochs.
    ############################
    if ((e+1) % 100 == 0):
        learning_rate = learning_rate / 2.0
    h = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    c = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    for i in range(num_batches):
        data_one_hot = train_data[i]
        label_one_hot = train_label[i]
        with autograd.record():
            outputs, h, c = lstm_rnn(data_one_hot, h, c)
            loss = average_ce_loss(outputs, label_one_hot)
            loss.backward()
        SGD(params, learning_rate)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        if (i == 0) and (e == 0):
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
      
    print("Epoch %s. Loss: %s" % (e, moving_loss)) 
    print(sample("The Time Ma", 1024, temperature=.1))
    print(sample("The Medical Man rose, came to the lamp,", 1024, temperature=.1))
            
```

## Conclusions

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
