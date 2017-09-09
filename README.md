# Tree-based Convolutional Neural Networks

Efficient TBCNN implemented in TensorFlow based on the following paper:

["Convolutional Neural Networks over Tree Structures for Programming Language Processing" Lili Mou, et al.](https://arxiv.org/pdf/1409.5718.pdf)

Differences from the paper
--------------------------

* There is no "coding layer" in this implementation (works fine without it).
* Vectors are learned by a variation of word2vec instead of the proposed method.
* Adam is used instead of gradient descent.

Usage
=====

Recommended process is to setup a Python Virtual environment.

Usage notes
-----------

While every attempt has been made to make this model memory efficient, it is
capable of accepting arbitrary-sized trees. Especially large trees can consume
many gigabytes of memory, so it is suggested you work with only small trees.
You can filter out large trees in the sampling step with the --maxsize flag.
Depending on your machine's setup you can grow or shrink this number. The model
can run fairly well (albeit slower) on a CPU with RAM if memory is a problem.

The vectorizer has best results with large amounts of data. The sample data 
source provided is fairly small. It is possible to train the vectorizer on
a larger datasource, and then apply it to a smaller classification problem.

First time setup
----------------

This will create a Python virtual environment, and install the necessary
packages only for this project.

    $ pip install virtualenv
    $ virtualenv -p /usr/bin/python2 venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    $ python setup.py develop

Note the recommended Python version is 2 because many of the scripts parsed
by the AST parser are written in Python 2. (Although we would use Python 3
if we could.)

Running from the virtual environment
------------------------------------

Make sure you run the commands from inside the virtual environment, once the
virtual environment is created, you can enter it with:

    $ source venv/bin/activate

Crawler
-------

Create a GitHub access token: https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/

    $ cp crawler/config.sample.json crawler/config.json
    $ (vim|emacs|nano) crawler/config.json

Add your user name and access token to the config file.

To download algorithm data from GitHub and parse the syntax trees into an output file:

    $ mkdir crawler/data
    $ crawl algorithms --out crawler/data/algorithms.pkl
    
**Alternatively:** a sample dataset fetched in the same way is saved in crawler/algorithms.zip. You can simply unzip it into the crawler/data directory.

Vectorizer
----------

Sample AST nodes from the GitHub data and output the sampled nodes into a file:

    $ mkdir sampler/data
    $ sample nodes --in  crawler/data/algorithms.pkl \
                   --out sampler/data/algorithm_nodes.pkl

Turn the sampled nodes into a vector embedding and output the embedding to a
file:

    $ mkdir vectorizer/data
    $ vectorize ast2vec --in         sampler/data/algorithm_nodes.pkl \
                        --out        vectorizer/data/vectors.pkl \
                        --checkpoint vectorizer/logs/algorithms

Visualize vectorizer embeddings using TensorBoard:

    $ tensorboard --logdir=vectorizer/logs/algorithms

Classifier
----------

To sample small trees with a 70/30 train/test split:

    $ sample trees --in      crawler/data/algorithms.pkl \
                   --out     sampler/data/algorithm_trees.pkl \
                   --maxsize 2000 \
                   --test    30

To train the classifier:

    $ classify train tbcnn --in    sampler/data/algorithm_trees.pkl \
                           --logdir classifier/logs/1 \
                           --embed  vectorizer/data/vectors.pkl

Test the classifier results

    $ classify test tbcnn --in     sampler/data/algorithm_trees.pkl \
                          --logdir classifier/logs/1 \
                          --embed  vectorizer/data/vectors.pkl

Example output
--------------

The classification task was to classify 6 different kinds of data structures
and argorithms.

### After training

    ('Accuracy:', 0.9924924924924925)
                precision    recall  f1-score   support

      mergesort       1.00      1.00      1.00       413
     linkedlist       1.00      1.00      1.00       368
      quicksort       1.00      1.00      1.00       401
            bfs       0.95      1.00      0.98       313
     bubblesort       1.00      1.00      1.00       185
       knapsack       1.00      0.95      0.98       318

    avg / total       0.99      0.99      0.99      1998

    [[413   0   0   0   0   0]
     [  0 368   0   0   0   0]
     [  0   0 401   0   0   0]
     [  0   0   0 313   0   0]
     [  0   0   0   0 185   0]
     [  0   0   0  15   0 303]]


### After testing

    ('Accuracy:', 0.99300699300699302)
                precision    recall  f1-score   support

      mergesort       1.00      1.00      1.00       154
     linkedlist       1.00      1.00      1.00       157
      quicksort       1.00      1.00      1.00       166
            bfs       0.96      1.00      0.98       128
     bubblesort       1.00      1.00      1.00       109
       knapsack       1.00      0.96      0.98       144

    avg / total       0.99      0.99      0.99       858

    [[154   0   0   0   0   0]
     [  0 157   0   0   0   0]
     [  0   0 166   0   0   0]
     [  0   0   0 128   0   0]
     [  0   0   0   0 109   0]
     [  0   0   0   6   0 138]]

Acknowledgements
----------------

Thanks to [Zhao Yan](https://github.com/GuitarmonYz) for helping figure out every
problem.
