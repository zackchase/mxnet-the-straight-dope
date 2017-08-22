# Tree LSTM modeling for semantic relatedness

### Sentences involving Compositional Knowledge
This tutorial walks through training a child-sum Tree LSTM model for analyzing semantic relatedness of sentence pairs given their dependency parse trees.

### Preliminaries
Requires the latest MXNet with the new `gluon` interface. One can either build from source or install the pre-release package through `pip install --pre mxnet`. Use of GPUs is preferred if one wants to run the complete training to match the state-of-the-art results.

Besides, to show a progress meter, one should install the `tqdm` ("progress" in Arabic) through  `pip install tqdm`. One should also install the HTTP library through `pip install requests`.


### Inspiration
This tutorial borrows heavily from [Pytorch](https://github.com/dasguptar/treelstm.pytorch) example.

```{.python .input  n=1}
import mxnet as mx
from mxnet.gluon import Block, nn
from mxnet.gluon.parameter import Parameter
```

```{.python .input  n=2}
class Tree(object):
    def __init__(self, idx):
        self.children = []
        self.idx = idx

    def __repr__(self):
        if self.children:
            return '{0}: {1}'.format(self.idx, str(self.children))
        else:
            return str(self.idx)
```

```{.python .input  n=3}
tree = Tree(0)
tree.children.append(Tree(1))
tree.children.append(Tree(2))
tree.children.append(Tree(3))
tree.children[1].children.append(Tree(4))
print(tree)
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0: [1, 2: [4], 3]\n"
 }
]
```

###  Model
The model is based on [child-sum tree LSTM](https://nlp.stanford.edu/pubs/tai-socher-manning-acl2015.pdf). For each sentence, the tree LSTM model extracts information following the dependency parse tree structure, and produces the sentence embedding at the root of each tree. This embedding can be used to predict semantic similarity.

#### Child-sum Tree LSTM

```{.python .input  n=4}
class ChildSumLSTMCell(Block):
    def __init__(self, hidden_size,
                 i2h_weight_initializer=None,
                 hs2h_weight_initializer=None,
                 hc2h_weight_initializer=None,
                 i2h_bias_initializer='zeros',
                 hs2h_bias_initializer='zeros',
                 hc2h_bias_initializer='zeros',
                 input_size=0, prefix=None, params=None):
        super(ChildSumLSTMCell, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._hidden_size = hidden_size
            self._input_size = input_size
            self.i2h_weight = self.params.get('i2h_weight', shape=(4*hidden_size, input_size),
                                              init=i2h_weight_initializer)
            self.hs2h_weight = self.params.get('hs2h_weight', shape=(3*hidden_size, hidden_size),
                                               init=hs2h_weight_initializer)
            self.hc2h_weight = self.params.get('hc2h_weight', shape=(hidden_size, hidden_size),
                                               init=hc2h_weight_initializer)
            self.i2h_bias = self.params.get('i2h_bias', shape=(4*hidden_size,),
                                            init=i2h_bias_initializer)
            self.hs2h_bias = self.params.get('hs2h_bias', shape=(3*hidden_size,),
                                             init=hs2h_bias_initializer)
            self.hc2h_bias = self.params.get('hc2h_bias', shape=(hidden_size,),
                                             init=hc2h_bias_initializer)

    def forward(self, F, inputs, tree):
        children_outputs = [self.forward(F, inputs, child)
                            for child in tree.children]
        if children_outputs:
            _, children_states = zip(*children_outputs) # unzip
        else:
            children_states = None

        with inputs.context as ctx:
            return self.node_forward(F, F.expand_dims(inputs[tree.idx], axis=0), children_states,
                                     self.i2h_weight.data(ctx),
                                     self.hs2h_weight.data(ctx),
                                     self.hc2h_weight.data(ctx),
                                     self.i2h_bias.data(ctx),
                                     self.hs2h_bias.data(ctx),
                                     self.hc2h_bias.data(ctx))

    def node_forward(self, F, inputs, children_states,
                     i2h_weight, hs2h_weight, hc2h_weight,
                     i2h_bias, hs2h_bias, hc2h_bias):
        # comment notation:
        # N for batch size
        # C for hidden state dimensions
        # K for number of children.

        # FC for i, f, u, o gates (N, 4*C), from input to hidden
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=i2h_bias,
                               num_hidden=self._hidden_size*4)
        i2h_slices = F.split(i2h, num_outputs=4) # (N, C)*4
        i2h_iuo = F.concat(*[i2h_slices[i] for i in [0, 2, 3]], dim=1) # (N, C*3)

        if children_states:
            # sum of children states, (N, C)
            hs = F.add_n(*[state[0] for state in children_states])
            # concatenation of children hidden states, (N, K, C)
            hc = F.concat(*[F.expand_dims(state[0], axis=1) for state in children_states], dim=1)
            # concatenation of children cell states, (N, K, C)
            cs = F.concat(*[F.expand_dims(state[1], axis=1) for state in children_states], dim=1)

            # calculate activation for forget gate. addition in f_act is done with broadcast
            i2h_f_slice = i2h_slices[1]
            f_act = i2h_f_slice + hc2h_bias + F.dot(hc, hc2h_weight) # (N, K, C)
            forget_gates = F.Activation(f_act, act_type='sigmoid') # (N, K, C)
        else:
            # for leaf nodes, summation of children hidden states are zeros.
            hs = F.zeros_like(i2h_slices[0])

        # FC for i, u, o gates, from summation of children states to hidden state
        hs2h_iuo = F.FullyConnected(data=hs, weight=hs2h_weight, bias=hs2h_bias,
                                    num_hidden=self._hidden_size*3)
        i2h_iuo = i2h_iuo + hs2h_iuo

        iuo_act_slices = F.SliceChannel(i2h_iuo, num_outputs=3) # (N, C)*3
        i_act, u_act, o_act = iuo_act_slices[0], iuo_act_slices[1], iuo_act_slices[2] # (N, C) each

        # calculate gate outputs
        in_gate = F.Activation(i_act, act_type='sigmoid')
        in_transform = F.Activation(u_act, act_type='tanh')
        out_gate = F.Activation(o_act, act_type='sigmoid')

        # calculate cell state and hidden state
        next_c = in_gate * in_transform
        if children_states:
            next_c = F.sum(forget_gates * cs, axis=1) + next_c
        next_h = out_gate * F.Activation(next_c, act_type='tanh')

        return next_h, [next_h, next_c]
```

#### Similarity regression module

```{.python .input  n=5}
# module for distance-angle similarity
class Similarity(nn.Block):
    def __init__(self, sim_hidden_size, rnn_hidden_size, num_classes):
        super(Similarity, self).__init__()
        with self.name_scope():
            self.wh = nn.Dense(sim_hidden_size, in_units=2*rnn_hidden_size)
            self.wp = nn.Dense(num_classes, in_units=sim_hidden_size)

    def forward(self, F, lvec, rvec):
        # lvec and rvec will be tree_lstm cell states at roots
        mult_dist = F.broadcast_mul(lvec, rvec)
        abs_dist = F.abs(F.add(lvec,-rvec))
        vec_dist = F.concat(*[mult_dist, abs_dist],dim=1)
        out = F.log_softmax(self.wp(F.sigmoid(self.wh(vec_dist))))
        return out
```

#### Final model

```{.python .input  n=6}
# putting the whole model together
class SimilarityTreeLSTM(nn.Block):
    def __init__(self, sim_hidden_size, rnn_hidden_size, embed_in_size, embed_dim, num_classes):
        super(SimilarityTreeLSTM, self).__init__()
        with self.name_scope():
            self.embed = nn.Embedding(embed_in_size, embed_dim)
            self.childsumtreelstm = ChildSumLSTMCell(rnn_hidden_size, input_size=embed_dim)
            self.similarity = Similarity(sim_hidden_size, rnn_hidden_size, num_classes)

    def forward(self, F, l_inputs, r_inputs, l_tree, r_tree):
        l_inputs = self.embed(l_inputs)
        r_inputs = self.embed(r_inputs)
        # get cell states at roots
        lstate = self.childsumtreelstm(F, l_inputs, l_tree)[1][1]
        rstate = self.childsumtreelstm(F, r_inputs, r_tree)[1][1]
        output = self.similarity(F, lstate, rstate)
        return output
```

### Dataset classes
#### Vocab

```{.python .input  n=7}
import os
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import random
from tqdm import tqdm

import mxnet as mx

# class for vocabulary and the word embeddings
class Vocab(object):
    # constants for special tokens: padding, unknown, and beginning/end of sentence.
    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD = '<blank>', '<unk>', '<s>', '</s>'
    def __init__(self, filepaths=[], embedpath=None, include_unseen=False, lower=False):
        self.idx2tok = []
        self.tok2idx = {}
        self.lower = lower
        self.include_unseen = include_unseen

        self.add(Vocab.PAD_WORD)
        self.add(Vocab.UNK_WORD)
        self.add(Vocab.BOS_WORD)
        self.add(Vocab.EOS_WORD)

        self.embed = None

        for filename in filepaths:
            logging.info('loading %s'%filename)
            with open(filename, 'r') as f:
                self.load_file(f)
        if embedpath is not None:
            logging.info('loading %s'%embedpath)
            with open(embedpath, 'r') as f:
                self.load_embedding(f, reset=set([Vocab.PAD_WORD, Vocab.UNK_WORD, Vocab.BOS_WORD,
                                                  Vocab.EOS_WORD]))

    @property
    def size(self):
        return len(self.idx2tok)

    def get_index(self, key):
        return self.tok2idx.get(key.lower() if self.lower else key,
                                Vocab.UNK)

    def get_token(self, idx):
        if idx < self.size:
            return self.idx2tok[idx]
        else:
            return Vocab.UNK_WORD

    def add(self, token):
        token = token.lower() if self.lower else token
        if token in self.tok2idx:
            idx = self.tok2idx[token]
        else:
            idx = len(self.idx2tok)
            self.idx2tok.append(token)
            self.tok2idx[token] = idx
        return idx

    def to_indices(self, tokens, add_bos=False, add_eos=False):
        vec = [BOS] if add_bos else []
        vec += [self.get_index(token) for token in tokens]
        if add_eos:
            vec.append(EOS)
        return vec

    def to_tokens(self, indices, stop):
        tokens = []
        for i in indices:
            tokens += [self.get_token(i)]
            if i == stop:
                break
        return tokens

    def load_file(self, f):
        for line in f:
            tokens = line.rstrip('\n').split()
            for token in tokens:
                self.add(token)

    def load_embedding(self, f, reset=[]):
        vectors = {}
        for line in tqdm(f.readlines(), desc='Loading embeddings'):
            tokens = line.rstrip('\n').split(' ')
            word = tokens[0].lower() if self.lower else tokens[0]
            if self.include_unseen:
                self.add(word)
            if word in self.tok2idx:
                vectors[word] = [float(x) for x in tokens[1:]]
        dim = len(vectors.values()[0])
        def to_vector(tok):
            if tok in vectors and tok not in reset:
                return vectors[tok]
            elif tok not in vectors:
                return np.random.normal(-0.05, 0.05, size=dim)
            else:
                return [0.0]*dim
        self.embed = mx.nd.array([vectors[tok] if tok in vectors and tok not in reset
                                  else [0.0]*dim for tok in self.idx2tok])
```

#### Data iterator

```{.python .input  n=8}
# Iterator class for SICK dataset
class SICKDataIter(object):
    def __init__(self, path, vocab, num_classes, shuffle=True):
        super(SICKDataIter, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes
        self.l_sentences = []
        self.r_sentences = []
        self.l_trees = []
        self.r_trees = []
        self.labels = []
        self.size = 0
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        if self.shuffle:
            mask = list(range(self.size))
            random.shuffle(mask)
            self.l_sentences = [self.l_sentences[i] for i in mask]
            self.r_sentences = [self.r_sentences[i] for i in mask]
            self.l_trees = [self.l_trees[i] for i in mask]
            self.r_trees = [self.r_trees[i] for i in mask]
            self.labels = [self.labels[i] for i in mask]
        self.index = 0

    def next(self):
        out = self[self.index]
        self.index += 1
        return out

    def set_context(self, context):
        self.l_sentences = [a.as_in_context(context) for a in self.l_sentences]
        self.r_sentences = [a.as_in_context(context) for a in self.r_sentences]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        l_tree = self.l_trees[index]
        r_tree = self.r_trees[index]
        l_sent = self.l_sentences[index]
        r_sent = self.r_sentences[index]
        label = self.labels[index]
        return (l_tree, l_sent, r_tree, r_sent, label)
```

### Training with autograd

```{.python .input  n=9}
import argparse, pickle, math, os, random
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag

# training settings and hyper-parameters
use_gpu = False
optimizer = 'AdaGrad'
seed = 123
batch_size = 25
training_batches_per_epoch = 10
learning_rate = 0.01
weight_decay = 0.0001
epochs = 1
rnn_hidden_size, sim_hidden_size, num_classes = 150, 50, 5

# initialization
context = [mx.gpu(0) if use_gpu else mx.cpu()]

# seeding
mx.random.seed(seed)
np.random.seed(seed)
random.seed(seed)

# read dataset
def verified(file_path, sha1hash):
    import hashlib
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    matched = sha1.hexdigest() == sha1hash
    if not matched:
        logging.warn('Found hash mismatch in file {}, possibly due to incomplete download.'
                     .format(file_path))
    return matched

data_file_name = 'tree_lstm_dataset-3d85a6c4.cPickle'
data_file_hash = '3d85a6c44a335a33edc060028f91395ab0dcf601'
if not os.path.exists(data_file_name) or not verified(data_file_name, data_file_hash):
    from mxnet.test_utils import download
    download('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/%s'%data_file_name,
             overwrite=True)


with open('tree_lstm_dataset-3d85a6c4.cPickle', 'rb') as f:
    train_iter, dev_iter, test_iter, vocab = pickle.load(f)

logging.info('==> SICK vocabulary size : %d ' % vocab.size)
logging.info('==> Size of train data   : %d ' % len(train_iter))
logging.info('==> Size of dev data     : %d ' % len(dev_iter))
logging.info('==> Size of test data    : %d ' % len(test_iter))

# get network
net = SimilarityTreeLSTM(sim_hidden_size, rnn_hidden_size, vocab.size, vocab.embed.shape[1], num_classes)

# use pearson correlation and mean-square error for evaluation
metric = mx.metric.create(['pearsonr', 'mse'])

# the prediction from the network is log-probability vector of each score class
# so use the following function to convert scalar score to the vector
# e.g 4.5 -> [0, 0, 0, 0.5, 0.5]
def to_target(x):
    target = np.zeros((1, num_classes))
    ceil = int(math.ceil(x))
    floor = int(math.floor(x))
    if ceil==floor:
        target[0][floor-1] = 1
    else:
        target[0][floor-1] = ceil - x
        target[0][ceil-1] = x - floor
    return mx.nd.array(target)

# and use the following to convert log-probability vector to score
def to_score(x):
    levels = mx.nd.arange(1, 6, ctx=x.context)
    return [mx.nd.sum(levels*mx.nd.exp(x), axis=1).reshape((-1,1))]

# when evaluating in validation mode, check and see if pearson-r is improved
# if so, checkpoint and run evaluation on test dataset
def test(ctx, data_iter, best, mode='validation', num_iter=-1):
    data_iter.reset()
    samples = len(data_iter)
    data_iter.set_context(ctx[0])
    preds = []
    labels = [mx.nd.array(data_iter.labels, ctx=ctx[0]).reshape((-1,1))]
    for _ in tqdm(range(samples), desc='Testing in {} mode'.format(mode)):
        l_tree, l_sent, r_tree, r_sent, label = data_iter.next()
        z = net(mx.nd, l_sent, r_sent, l_tree, r_tree)
        preds.append(z)

    preds = to_score(mx.nd.concat(*preds, dim=0))
    metric.update(preds, labels)
    names, values = metric.get()
    metric.reset()
    for name, acc in zip(names, values):
        logging.info(mode+' acc: %s=%f'%(name, acc))
        if name == 'pearsonr':
            test_r = acc
    if mode == 'validation' and num_iter >= 0:
        if test_r >= best:
            best = test_r
            logging.info('New optimum found: {}.'.format(best))
        return best


def train(epoch, ctx, train_data, dev_data):
    # initialization with context
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx[0])
    net.embed.weight.set_data(vocab.embed.as_in_context(ctx[0]))
    train_data.set_context(ctx[0])
    dev_data.set_context(ctx[0])

    # set up trainer for optimizing the network.
    trainer = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': learning_rate, 'wd': weight_decay})

    best_r = -1
    Loss = gluon.loss.KLDivLoss()
    for i in range(epoch):
        train_data.reset()
        num_samples = min(len(train_data), training_batches_per_epoch*batch_size)
        # collect predictions and labels for evaluation metrics
        preds = []
        labels = [mx.nd.array(train_data.labels[:num_samples], ctx=ctx[0]).reshape((-1,1))]
        for j in tqdm(range(num_samples), desc='Training epoch {}'.format(i)):
            # get next batch
            l_tree, l_sent, r_tree, r_sent, label = train_data.next()
            # use autograd to record the forward calculation
            with ag.record():
                # forward calculation. the output is log probability
                z = net(mx.nd, l_sent, r_sent, l_tree, r_tree)
                # calculate loss
                loss = Loss(z, to_target(label).as_in_context(ctx[0]))
                # backward calculation for gradients.
                loss.backward()
                preds.append(z)
            # update weight after every batch_size samples
            if (j+1) % batch_size == 0:
                trainer.step(batch_size)

        # translate log-probability to scores, and evaluate
        preds = to_score(mx.nd.concat(*preds, dim=0))
        metric.update(preds, labels)
        names, values = metric.get()
        metric.reset()
        for name, acc in zip(names, values):
            logging.info('training acc at epoch %d: %s=%f'%(i, name, acc))
        best_r = test(ctx, dev_data, best_r, num_iter=i)

train(epochs, context, train_iter, dev_iter)
```

```{.json .output n=9}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "INFO:root:==> SICK vocabulary size : 2412 \nINFO:root:==> Size of train data   : 4500 \nINFO:root:==> Size of dev data     : 500 \nINFO:root:==> Size of test data    : 4927 \nTraining epoch 0: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 250/250 [00:11<00:00, 21.48it/s]\nINFO:root:training acc at epoch 0: pearsonr=0.096197\nINFO:root:training acc at epoch 0: mse=1.138699\nTesting in validation mode: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 500/500 [00:09<00:00, 51.57it/s]\nINFO:root:validation acc: pearsonr=0.490352\nINFO:root:validation acc: mse=1.237509\nINFO:root:New optimum found: 0.49035187610029013.\n"
 }
]
```

### Conclusion
- Gluon offers great tools for modeling in an imperative way.
