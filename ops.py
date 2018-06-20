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

    
def batch_norm(x, phase, name='batch_norm'):
    if phase is 'train':
        phase = True
    else:
        phase = False
    return tf.contrib.layers.batch_norm(x, decay=0.9, is_training=phase, 
                                        epsilon=1e-5, center=True, scale=True, scope=name)


def dropout(x, rate, phase):
    if phase is 'train':
        phase = True
    else:
        phase = False
    return tf.layers.dropout(inputs=x, rate=rate, training=phase)


def average_pooling(x, ks=[2,2], s=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=ks, strides=s, padding=padding)


def max_pooling(x, ks=[3,3], s=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=ks, strides=s, padding=padding)

def flatten(x):
    return tf.contrib.layers.flatten(x)

def relu(x, name='relu'):
    return tf.nn.relu(x)

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def tanh(x):
    return tf.nn.tanh(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)