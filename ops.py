import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv2d(input_, output_dim, ks=3,s=1,padding='SAME',name='conv2d'):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding,
                           weights_initializer=tf.contrib.layers.xavier_initializer())
     
def linear(input_, output_dim, stddev=0.02, name='linear'):
    with tf.variable_scope(name):
        return slim.fully_connected(input_, output_dim, activation_fn=None,
                                    weights_initializer=tf.contrib.layers.xavier_initializer())
    
# def maxpool2d(input_, ks=2, s=2, padding='VALID',name='maxpool2d'):
#     with tf.variable_scope(name):
#         return slim.max_pool2d(input_, ks, s, padding=padding)

def batch_norm(x, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def relu(x, name='relu'):
    return tf.nn.relu(x)

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def tanh(x):
    return tf.nn.tanh(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)