## Introduction








Machine Learning is a versatile set of tools that lets you work with data in a
way where simple rule-based systems would fail or be very difficult to build.
Due to its versatility, machine learning can seem quite confusing in its
manifestations (search engines, self driving cars, machine translation, medical
diagnosis, spam filtering, chess and go playing, face recognition, finding a
date, calculating insurance premiums, fancy filters for photos, etc.). This is
largely due to the fact that machine learning lets you *program with data*
rather than having to write code to accomplish a task.

The language to express such things is *math*. We'll keep the latter to a
minimum and focus mostly on code. That said, it's probably a good idea to know
what a derivative (aka gradient) is, and what a matrix-vector product does.
We'll be gentle and explain it as we go along. To get a better idea of what
machine learning can do, let's consider a few examples.

## Programming with Code vs. Programming with Data

This example is inspired by an interaction that [Joel Grus](http://joelgrus.com)
had in a [job interview](http://joelgrus.com/2016/05/23/fizz-buzz-in-
tensorflow/). The interviewer asked him to code up Fizz Buzz. This is a
children's game where the players count from 1 to 100 and will say *'fizz'*
whenever the number is divisible by 3, *'buzz'* whenever it is divisible by 5,
and *'fizz buzz'* whenever it satisfies both criteria. Otherwise they will just
state the number. It looks like this:

``1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 fizz 13 14 fizzbuzz 16 ...``

The conventional way to solve such a task is quite simple.

```{.python .input  n=1}
for i in range(1,101):
    if (i % 3) == 0:
        print('fizz', end='')
    if (i % 5) == 0:
        print('buzz', end='')
    if (i % 3) * (i % 5):
        print(i, end='')
    print(' ', end='')
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 fizz 13 14 fizzbuzz 16 17 fizz 19 buzz fizz 22 23 fizz buzz 26 fizz 28 29 fizzbuzz 31 32 fizz 34 buzz fizz 37 38 fizz buzz 41 fizz 43 44 fizzbuzz 46 47 fizz 49 buzz fizz 52 53 fizz buzz 56 fizz 58 59 fizzbuzz 61 62 fizz 64 buzz fizz 67 68 fizz buzz 71 fizz 73 74 fizzbuzz 76 77 fizz 79 buzz fizz 82 83 fizz buzz 86 fizz 88 89 fizzbuzz 91 92 fizz 94 buzz fizz 97 98 fizz buzz "
 }
]
```

Needless to say, this isn't very exciting if you're a good programmer. Joel
proceeded to 'implement' this problem in Machine Learning instead. For that to
succeed, he needed a number of pieces:

* Data X ``[1, 2, 3, 4, ...]`` and labels Y ``['fizz', 'buzz', 'fizzbuzz',
identity]``
* Training data, i.e. examples of what the system is supposed to do. Such as
``[(2, 2), (6, fizz), (15, fizzbuzz), (23, 23), (40, buzz)]``
* Features the map the data into something that the computer can handle more
easily, e.g. ``x -> [(x % 3), (x % 5), (x % 15)]``. This is optional but helps a
lot if you have it.

Armed with this, Joel wrote a classifier in TensorFlow
([code](https://github.com/joelgrus/fizz-buzz-tensorflow)). The interviewer was
nonplussed ... and the classifier didn't have perfect accuracy.

Quite obviously, this is silly. Why would you go through the trouble of
replacing a few lines of Python with something much more complicated and error
prone. However, there are many cases where a simple Python script simply does
not exist, yet a 3 year old child will solve the problem perfectly.

|![](img/cat1.jpg)|![](img/cat2.jpg)|![](img/dog1.jpg)|![](img/dog2.jpg)|
|:---------------:|:---------------:|:---------------:|:---------------:|
|cat|cat|dog|dog|

Fortunately, this is precisely where machine learning comes to the rescue. We
can 'program' a cat detector by providing our machine learning system with many
examples of cats and dogs. This way it will eventually learn a function that
will e.g. emit a very large positive number if it's a cat, a very large negative
number if it's a dog, and something closer to zero if it isn't sure. But this is
just barely scratching the surface of what machine learning can do ...

## Bare Bones Machine Learning

To succeed at machine learning, we need four things: data, a model of how to
transform the data, a loss function to measure how well we're doing, and an
algorithm to tweak the model parameters such that the loss function is
minimized.

* **Data.** The more the better. Data is at the heart of the resurgence of deep
learning, since complex nonlinear models require lots of it to work well. Here
are some examples.
  * Images, e.g. pictures taken by your phone (cats, dogs, dinosaurs, high
school reunions, dentist's X-Ray images, satellite images, ...)
  * Text, e.g. emails, high school essays, tweets, news articles
  * Sound, e.g. voice commands sent to an Echo, audio books, phone calls
  * Video, e.g. movies, home surveillance, multi-camera tracking, ...
  * Structured data, e.g. this Jupyter notebook (it contains text, images,
code), a webpage, car rental records, your electricity bill, ...
* **Model.** Usually the data looks quite different from what we want to
accomplish with it. E.g. we might have photos of people and want to know whether
they're happy. Hence, we need to turn a 10 megapixel image into a probability of
happiness. For this, we need to apply a number of (typically) nonlinear
transformations $f$, e.g. by defining a network.
* **Loss function.** To assess how well we're doing we need to compare the
output from the model with the truth. Loss functions allow us to determine
whether a stock prediction of \$1,500 for ``AMZN`` by December 31, 2017 is
correct. Depending on whether we decided to go short or long on it, we would
incur different losses (or realize profits), hence our loss functions might be
quite different.
* **Training.** Usually models have many parameters. These are the ones that we
need to 'learn', by minimizing the loss incurred on training data.
Unfortunately, doing well on the latter doesn't guarantee that we will do well
on (unseen) test data, as the analogy below illustrates.
  * Training Error - This is the error on the dataset used to find f by
minimizing the loss on the training set. This is equivalent to doing well on all
the practice exams that a student might use to prepare for the real exam.
Encouraging but by no means a guarantee.
  * Test Error - This is the error incurred on an unseen test set. This can be
off by quite a bit (statisticians call this overfitting). In real-life terms
this is the equivalent of screwing up the real exam despite doing well on the
practice exams.

In the following we will discuss a few types of machine learning in some more
detail. This helps to understand what exactly one aims to do.

## Classification

This is one of the simplest tasks. Given data $x \in X$, such as images, text,
sound, video, medical diagnostics, performance of a car, motion sensor data,
etc., we want to answer the question as to which class $y \in Y$ the data
belongs to. In the above case, $X$ are images and $Y = \mathrm{\{cat, dog\}}$.
Quite often the confidence of the classifier, i.e. the algorithm that does this,
is expressed in the form of probabilities, e.g. $\Pr(y=\mathrm{cat}|x) = 0.9$,
i.e. the classifier is 90% sure that it's a cat. Whenever we have only two
possible outcomes, statisticians call this a *binary classifier*. All other
cases are called *multiclass classification*, e.g. the digits ``[0, 1, 2, 3 ...
9]`` in a digit recognition task. In ``MxNet Gluon`` the corresponding loss
function is the [Cross Entropy Loss](http://mxnet.io/api/python/gluon.html#mxnet
.gluon.loss.SoftmaxCrossEntropyLoss).

Note that the most likely class is not necessarily the one that you're going to
use for your decision. Assume that you find this beautiful mushroom in your
backyard:

|![](img/death_cap.jpg)|
|:-------:|
|Death cap - do not eat!|

Our (admittedly quite foolish) classifier outputs $\Pr(y=\mathrm{death
cap}|\mathrm{image}) = 0.2$. In other words, it is quite confident that it
*isn't* a death cap. Nonetheless, you shouldn't eat it, since the cost of eating
it (you die) is much higher than the cost of discarding it (you miss a tasty
meal). Let's check how this looks like in math:

$$L(a|x) = \mathbf{E}_{y \sim p(y|x)}[\mathrm{loss}(a,y)]$$

Hence, the loss $L$ incurred by eating the mushroom is $L(a=\mathrm{eat}|x) =
0.2 * \infty + 0.8 * 0 = \infty$, whereas the cost of discarding it is
$L(a=\mathrm{discard}|x) = 0.2 * 0 + 0.8 * 1 = 0.8$. We got lucky - as any
botanist would tell us, the above actually *is* a death cap.

There are way more fancy classification problems than the ones above. For
instance, we might have hierarchies. One of the first examples of such a thing
are due to Linnaeus, who applied it to animals. Usually this is referred to as
*hierarchical classification*. Typically the cost of misclassification will
depend on how far you've strayed from the truth, e.g. mistaking a poodle for a
schnautzer is no big deal but mistaking it for a dinosaur would be embarrassing.
On the other hand, mistaking a rattle snake for a garter snake could be deadly.
In other words, the cost might be *nonuniform* over the hierarchy of classes but
tends to increase the further away you are from the truth.

![](img/taxonomy.jpg)

## Tagging

It is worth noting that many problems are *not* classification problems.
Discerning cats and dogs by computer vision is relatively easy, but what should
our poor classifier do in this situation?

![](img/catdog.jpg)

Obviously there's a cat in the picture. And a dog. And a tire, some grass, a
door, concrete, rust, individual grass leaves, etc.; Treating it as a binary
classification problem is asking for trouble. Our poor classifier will get
horribly confused if it needs to decide whether the image is one of two things,
if it is actually both.

The above example seems contrived but what about this case: a picture of a model
posing in front of a car at the beach. Each of the tags ``(woman, car, beach)``
would be true. In other words, there are situations where we have multiple tags
or attributes of what is contained in an object. Sometimes this is treated as a
lot of binary classification problems. But this is problematic, too, since there
are just so many tags (often hundreds of thousands or millions) that could
apply, e.g. ``(ham, green eggs, spam, grinch, ...)`` and we would have to
*check* all of them and to ensure that they are all accurate.

Suffice it to say, there are better ways of generating tags. For instance, we
could try to estimate the probability that $y$ is one of the tags in the set
$S_x$ of tags associated with $x$, i.e. $\Pr(y \in S_x|x)$. We will discuss them
later in this tutorial (with actual code). For now just remember that *tagging
is not classification*.

## Regression

Let's assume that you're having your drains repaired and the contractor spends
$x=3$ hours removing gunk from your sewage pipes. He then sends you a bill of $y
= 350\$ $. Your friend hires the same contractor for $x = 2$ hours and he gets a
bill of $y = 250\$ $. You can now both team up and perform a regression estimate
to identify the contractor's pricing structure: \$100 per hour plus \$50 to show
up at your house. That is, $f(x) = 100 \cdot x + 50$.

More generally, in regression we aim to obtain a real-valued number $y \in
\mathbb{R}$ based on data $x$. Here $x$ could be as simple as the number of
hours worked, or as complex as last week's news if we want to estimate the gain
in a share price. For most of the tutorial, we will be using one of two very
common losses, the [L1
loss](http://mxnet.io/api/python/gluon.html#mxnet.gluon.loss.L1Loss) where
$l(y,y') = \sum_i |y_i-y_i'|$ and the [L2
loss](http://mxnet.io/api/python/gluon.html#mxnet.gluon.loss.L2Loss) where
$l(y,y') = \sum_i (y_i - y_i')^2$. As we will see later, the $L_2$ loss
corresponds to the assumption that our data was corrupted by Gaussian Noise,
whereas the $L_1$ loss is very robust to malicious data corruption, albeit at
the expense of lower efficiency.

## Search and Ranking

One of the problems quite different from classifications is ranking. There the
goal is less to determine whether a particular page is relevant for a query, but
rather, which one of the plethora of search results should be displayed for the
user. That is, we really care about the ordering among the relevant search
results and our learning algorithm needs to produce ordered subsets of elements
from a larger set. In other words, if we are asked to produce the first 5
letters from the alphabet, there is a difference between returning ``A B C D E``
and ``C A B E D``. Even if the result set is the same, the ordering within the
set matters nonetheless.

A possible solution to this problem is to score every element in the set of
possible sets with a relevance score and then retrieve the top-rated elements.
[PageRank](https://en.wikipedia.org/wiki/PageRank) is an early example of such a
relevance score. One of the peculiarities is that it didn't depend on the actual
query. Instead, it simply helped to order the results that contained the query
terms. Nowadays search engines use machine learning and behavioral models to
obtain relevance scores.

![](img/mxnet_google.png)



## Recommender Systems

Quite closely related to search and ranking are recommender systems. The
problems are  similar insofar as the goal is to display a set of relevant items
to the user. The main difference is the emphasis on *personalization* to
specific users in the context of recommender systems. For instance, for movie
recommendation the result page for a SciFi fan and the result page for a
connoissoir of Woody Allen comedies might differ significantly.

Such problems occur, e.g. for movie, product or music recommendation. In some
cases customers will provide explicit details about how much they liked the
product (e.g. Amazon product reviews). In some other cases they might simply
provide feedback if they are dissatisfied with the result (skipping titles on a
playlist). Generally, such systems strive to estimate some score $y_{ij}$ as a
function of user $u_i$ and object $o_j$. The objects $o_j$ with the largest
scores $y_{ij}$ are then used as recommendation. Production systems are
considerably more advanced and take detailed user activity and item
characteristics into account when computing such scores. Below an example of the
books recommended for deep learning, based on the author's preferences.

![](img/deeplearning_amazon.png)


## Sequence Transformations

Some of the more exciting applications of machine learning are sequence
transformations, sometimes also referred as ``seq2seq`` problems. They ingest a
sequence of data and emit a new, significantly transformed one. There are many
use cases:

* **Tagging and Parsing.** This involves annotating a text sequence with
attributes. For instance, we might want to know where the verbs and subjects
are, we might want to know which words are the named entities. In general, the
goal is to decompose and annotate text $x$ based on structural and grammatical
assumptions to get some annotation $y$. This sounds more complex than it
actually is. Below is a very simple example.

```Tom wants to have dinner in Washington with Sally.
E   -     -  -    -      -  E          -    E```

* **Automatic Speech Recognition.** Here the input sequence $x$ is the sound of
a speaker, and the output $y$ is the textual transcript of what the speaker
said. The challenge is that there are many more audio frames (sound is typically
sampled at 8kHz or 16kHz), hence there is no 1:1 correspondence between audio
and text.

|----D----e--e--e-----p----------- L----ea-------r---------ni-----ng----- |
|--------------|
|![Deep Learning](img/speech.jpg)|

* **Text to Speech.** TTS is the inverse of Speech Recognition. That is, the
input $x$ is text and the output $y$ is an audio file. There, the output is
*much longer* than the input. While it is easy for *humans* to recognize a bad
audio file, this isn't quite so trivial for computers. The challenge is that the
audio output is way longer than the input sequence.

* **Machine Translation.** The goal here is to map text from one language
automatically to the other. In a way, this generalizes the simple tagging and
parsing problem since the order of words may no longer be preserved. We provide
both the correct translation and the one obtained by preserving word-order.

```De   : Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
En   : Did you already check out this excellent tutorial?
Wrong: Did you yourself already this excellent tutorial looked-at?```

There are many more related problems. For instance, the order in which a user
reads a webpage is a two-dimensional layout analysis problem. Likewise, for
dialog problems we need to take world-knowledge and prior state into account.
This is an active area of research.


## Unsupervised Learning

All the examples so far related to *Supervised Learning*, i.e. situations where
we know what we want. Quite often, though, we simply want to learn as much about
the data as possible. For instance, we might want to know whether we can
represent the data by a small number of prototypes (aka clustering), whethere
there is a low-dimensional subspace that captures most of the interesting pieces
of the data (principal component analysis and related autoencoders), or whether
there is some representation of the data that captures the attributes of the
data better (e.g. Rome - Italy + France = Paris). We will discuss this in a
chapter on unsupervised learning.
