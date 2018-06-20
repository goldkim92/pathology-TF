import tensorflow as tf
from ops import conv2d, linear, batch_norm, dropout, relu, max_pooling, average_pooling, flatten

def classifier(images, options, reuse=False, name='classifier'):
    # dropout_rate = 0.2
    x = relu(batch_norm(conv2d(images, options.nf, ks=5, s=2, name='conv1'), options.phase, 'bn1')) # 56*56*nf
    # x = dropout(x, dropout_rate, options.phase)
    x = relu(batch_norm(conv2d(x, 2*options.nf, ks=5, s=2, name='conv2'), options.phase, 'bn2')) # 28*28*(2*nf)
    # x = dropout(x, dropout_rate, options.phase)
    x = relu(batch_norm(conv2d(x, 4*options.nf, ks=5, s=2, name='conv3'), options.phase, 'bn3')) # 14*14*(4*nf)
    # x = dropout(x, dropout_rate, options.phase)
    x = relu(batch_norm(conv2d(x, 8*options.nf, ks=5, s=2, name='conv4'), options.phase, 'bn4')) # 7*7*(8*nf)
    
    x = linear(tf.reshape(x, [options.batch_size, 7*7*(8*options.nf)]), options.label_n, name='linear') 
    return tf.nn.softmax(x)


def cls_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))



''' 
** Dense Net **
This code is revised from 
https://github.com/taki0112/Densenet-Tensorflow/blob/master/MNIST/Densenet_MNIST.py
'''

class DenseNet():
    def __init__(self, x, nf, label_n, phase):
        self.nf = nf
        self.label_n = label_n
        self.phase = phase # 'train' or else
        self.dropout_rate = 0.2
        self.model = self.dense_net(x)


#     def conv_dropout(self, x, nf, ks=[3,3], s=1, scope=''):
#         x = batch_norm(x, self.phase, scope+'_batch1')
#         x = relu(x)
#         x = conv2d(x, nf, ks, s, name=scope+'_conv1')
#         x = dropout(x, self.dropout_rate, self.phase)
        
        
    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
#             x = conv_dropout(x, 4 * self.nf, ks=[3,3], s=1, scope)
            x = batch_norm(x, self.phase, scope+'_batch1')
            x = relu(x)
            x = conv2d(x, 2 * self.nf, ks=[1,1], s=1, name=scope+'_conv1')
#            x = dropout(x, self.dropout_rate, self.phase)

            
            x = batch_norm(x, self.phase, scope+'_batch2')
            x = relu(x)
            x = conv2d(x, self.nf, ks=[3,3], s=1, name=scope+'_conv2')
#            x = dropout(x, self.dropout_rate, self.phase)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = batch_norm(x, self.phase, scope+'_batch1')
            x = relu(x)
            x = conv2d(x, self.nf, ks=[1,1], s=1, name=scope+'_conv1')
#            x = dropout(x, self.dropout_rate, self.phase)
            
            x = average_pooling(x, ks=[2,2], s=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = tf.concat(layers_concat, axis=3)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = tf.concat(layers_concat, axis=3)

            return x

    def dense_net(self, input_x):
        x = conv2d(input_x, 2 * self.nf, ks=[7,7], s=2, name='conv0') # 56*56*(2*nf)
        print(x.get_shape().as_list())
        x = max_pooling(x, ks=[3,3], s=2, padding='SAME') # 28*28*(2*nf)
        print(x.get_shape().as_list())
        
        nb_blocks = 3
        for i in range(nb_blocks-1) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i)) 
        # 14*14*nf, 7*7*nf
        print(x.get_shape().as_list())

        """
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')
        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')
        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        """

        x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_final') # 4*4*((nb_layers+1)*nf)
        print(x.get_shape().as_list())
        x = batch_norm(x, self.phase, 'linear_batch')
        x = relu(x)
        x = average_pooling(x,ks=[7,7], s=1)
        print(x.get_shape().as_list())
        x = flatten(x)
        x = linear(x, self.label_n, name='linear2')
        return tf.nn.softmax(x)

        # 100 Layer
#         x = Batch_Normalization(x, training=self.training, scope='linear_batch')
#         x = Relu(x)
#         x = Global_Average_Pooling(x)
#         x = flatten(x)
#         x = Linear(x)


        # x = tf.reshape(x, [-1, 10])
        return x