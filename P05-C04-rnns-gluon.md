# Recurrent Neural Networks with ``gluon``


With gluon, now we can train the recurrent neural networks (RNNs) more neatly, such as the long short-term memory (LSTM) and the gated recurrent unit (GRU). To demonstrate the end-to-end RNN training and prediction pipeline, we take a classic problem in language modeling as a case study. Specifically, we will show how to predict the distribution of the next word given a sequence of previous words.

## Import packages

To begin with, we need to make the following necessary imports.

```{.python .input  n=1}
import math
import os
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
```

## Define classes for indexing words of the input document

In a language modeling problem, we define the following classes to facilitate the routine procedures for loading document data. In the following, the ``Dictionary`` class is for word indexing: words in the documents can be converted from the string format to the integer format. 

In this example, we use consecutive integers to index words of the input document.

```{.python .input  n=2}
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
```

The ``Dictionary`` class is used by the ``Corpus`` class to index the words of the input document.

```{.python .input  n=3}
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return mx.nd.array(ids, dtype='int32')
```

## Provide an exposition of different RNN models with ``gluon``

Based on the ``gluon.Block`` class, we can make different RNN models available with the following single ``RNNModel`` class.

Users can select their preferred RNN model or compare different RNN models by configuring the argument of the constructor of ``RNNModel``. We will show an example following the definition of the ``RNNModel`` class.

```{.python .input  n=4}
class RNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer = mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)
            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden,
                                        params = self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

## Select an RNN model and configure parameters

For demonstration purposes, we provide an arbitrary selection of the parameter values. In practice, some parameters should be more fine tuned based on the validation data set. 

For instance, to obtain a better performance, as reflected in a lower loss or perplexity, one can set ``args_epochs`` to a larger value.

In this demostration, LSTM is the chosen type of RNN. For other RNN options, one can replace the ``'lstm'`` string to ``'rnn_relu'``, ``'rnn_tanh'``, or ``'gru'`` as provided by the aforementioned ``gluon.Block`` class.

```{.python .input  n=5}
args_data = './data/nlp/ptb.'
args_model = 'rnn_relu'
args_emsize = 100
args_nhid = 100
args_nlayers = 2
args_lr = 1.0
args_clip = 0.2
args_epochs = 2
args_batch_size = 32
args_bptt = 5
args_dropout = 0.2
args_tied = True
args_cuda = 'store_true'
args_log_interval = 500
args_save = 'model.param'
```

## Load data as batches

We load the document data by leveraging the aforementioned ``Corpus`` class. 

To speed up the subsequent data flow in the RNN model, we pre-process the loaded data as batches. This procedure is defined in the following ``batchify`` function.

```{.python .input  n=6}
context = mx.cpu(0)
corpus = Corpus(args_data)

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

train_data = batchify(corpus.train, args_batch_size).as_in_context(context)
val_data = batchify(corpus.valid, args_batch_size).as_in_context(context)
test_data = batchify(corpus.test, args_batch_size).as_in_context(context)
```

## Build the model

We go on to build the model, initialize model parameters, and configure the optimization algorithms for training the RNN model.

```{.python .input  n=7}
ntokens = len(corpus.dictionary)

model = RNNModel(args_model, ntokens, args_emsize, args_nhid,
                       args_nlayers, args_dropout, args_tied)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Train the model and evaluate on validation and testing data sets

Now we can define functions for training and evaluating the model. The following are two helper functions that will be used during model training and evaluation.

```{.python .input  n=8}
def get_batch(source, i):
    seq_len = min(args_bptt, source.shape[0] - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden
```

The following is the function for model evaluation. It returns the loss of the model prediction. We will discuss the details of the loss measure shortly.

```{.python .input  n=9}
def eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, args_bptt):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal
```

Now we are ready to define the function for training the model. We can monitor the model performance on the training, validation, and testing data sets over iterations.

```{.python .input}
def train():
    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx = context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args_bptt)):
            data, target = get_batch(train_data, i)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, args_clip * args_bptt * args_batch_size)

            trainer.step(args_batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % args_log_interval == 0 and ibatch > 0:
                cur_L = total_L / args_bptt / args_batch_size / args_log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval(val_data)

        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            test_L = eval(test_data)
            model.save_params(args_save)
            print('test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
        else:
            args_lr = args_lr * 0.25
            trainer._init_optimizer('sgd',
                                    {'learning_rate': args_lr,
                                     'momentum': 0,
                                     'wd': 0})
            model.load_params(args_save, context)
```

Recall that the RNN model training is based on maximization likelihood of observations. For evaluation purposes, we have used the following two measures:

* Loss: the loss function is defined as the average negative log likelihood of the words under prediction: $$\text{loss} = -\frac{1}{N} \sum_{i = 1}^N \text{log} \  p_{\text{predicted}_i},  $$ where $N$ is the number of predictions and $p_{\text{predicted}_i}$ the likelihood of observing the next word in the $i$-th prediction.

* Perplexity: the average per-word perplexity is $\text{exp}(\text{loss})$.

To orient the reader using concrete examples, let us illustrate the idea of the perplexity measure as follows.

* Consider a perfect scenario where the prediction model always predicts the likelihood of the next word correctly. In this case, for every $i$ we have $p_{\text{predicted}_i} = 1$. As a result, the perplexity of a perfect prediction model is always 1. 

* Consider a baseline scenario where the prediction model always predicts the likelihood of the next word randomly at uniform among the given word set $W$. In this case, for every $i$ we have $p_{\text{predicted}_i} = 1 / |W|$. As a result, the perplexity of a uniformly random prediction model is always $|W|$. 

Therefore, a perplexity value is always between $1$ and $|W|$. A model with a lower perplexity that is closer to 1 is generally more accurate in prediction.

Now we are ready to train the model and evaluate the model performance on validation and testing data sets. 

```{.python .input}
train()
model.load_params(args_save, context)
test_L = eval(test_data)
print('Best test loss %.2f, test perplexity %.2f'%(test_L, math.exp(test_L)))
```

```{.json .output n=None}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[Epoch 1 Batch 500] loss 6.83, perplexity 926.78\n[Epoch 1 Batch 1000] loss 6.51, perplexity 670.66\n[Epoch 1 Batch 1500] loss 6.24, perplexity 511.26\n[Epoch 1 Batch 2000] loss 6.09, perplexity 442.18\n[Epoch 1 Batch 2500] loss 5.94, perplexity 379.94\n[Epoch 1 Batch 3000] loss 5.80, perplexity 329.72\n[Epoch 1 Batch 3500] loss 5.78, perplexity 324.35\n[Epoch 1 Batch 4000] loss 5.63, perplexity 278.87\n[Epoch 1 Batch 4500] loss 5.61, perplexity 272.14\n[Epoch 1 Batch 5000] loss 5.59, perplexity 268.51\n[Epoch 1 Batch 5500] loss 5.59, perplexity 266.44\n[Epoch 1] time cost 801.83s, validation loss 5.46, validation perplexity 234.71\ntest loss 5.43, test perplexity 227.80\n[Epoch 2 Batch 500] loss 5.54, perplexity 255.84\n[Epoch 2 Batch 1000] loss 5.48, perplexity 238.95\n[Epoch 2 Batch 1500] loss 5.43, perplexity 229.18\n[Epoch 2 Batch 2000] loss 5.46, perplexity 234.46\n[Epoch 2 Batch 2500] loss 5.40, perplexity 221.97\n[Epoch 2 Batch 3000] loss 5.32, perplexity 205.08\n[Epoch 2 Batch 3500] loss 5.35, perplexity 211.51\n[Epoch 2 Batch 4000] loss 5.25, perplexity 191.08\n[Epoch 2 Batch 4500] loss 5.25, perplexity 190.52\n[Epoch 2 Batch 5000] loss 5.28, perplexity 195.53\n[Epoch 2 Batch 5500] loss 5.29, perplexity 198.73\n[Epoch 2] time cost 827.04s, validation loss 5.21, validation perplexity 183.80\ntest loss 5.18, test perplexity 177.35\nBest test loss 5.18, test perplexity 177.35\n"
 }
]
```
