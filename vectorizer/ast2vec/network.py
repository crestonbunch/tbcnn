"""This creates a network for training a vector embedding of an AST using
a strategy similar to word2vec, but applied to the context of AST's."""

import math
import tensorflow as tf

from vectorizer.node_map import NODE_MAP
from vectorizer.ast2vec.parameters import \
    BATCH_SIZE, NUM_FEATURES, HIDDEN_NODES

def init_net(
        batch_size=BATCH_SIZE, num_feats=NUM_FEATURES, hidden_size=HIDDEN_NODES,
):
    """Construct the network graph."""

    with tf.name_scope('network'):

        with tf.name_scope('inputs'):
            # input node-child pairs
            inputs = tf.placeholder(tf.int32, shape=[batch_size,], name='inputs')
            labels = tf.placeholder(tf.int32, shape=[batch_size,], name='labels')

            # embeddings to learn
            embeddings = tf.Variable(
                tf.random_uniform([len(NODE_MAP), num_feats]), name='embeddings'
            )

            embed = tf.nn.embedding_lookup(embeddings, inputs)
            onehot_labels = tf.one_hot(labels, len(NODE_MAP), dtype=tf.float32)

        # weights will have features on the rows and nodes on the columns
        with tf.name_scope('hidden'):
            weights = tf.Variable(
                tf.truncated_normal(
                    [num_feats, hidden_size], stddev=1.0 / math.sqrt(num_feats)
                ),
                name='weights'
            )

            biases = tf.Variable(
                tf.zeros((hidden_size,)),
                name='biases'
            )

            hidden = tf.tanh(tf.matmul(embed, weights) + biases)

        with tf.name_scope('softmax'):
            weights = tf.Variable(
                tf.truncated_normal(
                    [hidden_size, len(NODE_MAP)],
                    stddev=1.0 / math.sqrt(hidden_size)
                ),
                name='weights'
            )
            biases = tf.Variable(
                tf.zeros((len(NODE_MAP),), name='biases')
            )

            logits = tf.matmul(hidden, weights) + biases

        with tf.name_scope('error'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=onehot_labels, logits=logits, name='cross_entropy'
            )

            loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    return inputs, labels, embeddings, loss
