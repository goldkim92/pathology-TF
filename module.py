import tensorflow as tf
from ops import conv2d, linear, batch_norm, relu
  


def classifier(images, options, reuse=False, name='classifier'):
    x = relu(batch_norm(conv2d(images, options.nf, ks=5, s=2, name='conv1'), 'bn1')) # 56*56*nf
    x = relu(batch_norm(conv2d(x, 2*options.nf, ks=5, s=2, name='conv2'), 'bn2')) # 28*28*(2*nf)
    x = relu(batch_norm(conv2d(x, 4*options.nf, ks=5, s=2, name='conv3'), 'bn3')) # 14*14*(4*nf)
    x = relu(batch_norm(conv2d(x, 8*options.nf, ks=5, s=2, name='conv4'), 'bn4')) # 7*7*(8*nf)
    
    x = linear(tf.reshape(x, [options.batch_size, 7*7*(8*options.nf)]), options.label_n, name='linear') 
    return tf.nn.softmax(x)


def cls_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))


# def densenet(image, options, reuse=False, name='densenet'):
    
#     divide = 2
    
#     h_conv1 = conv2d(image, options.nf, ks=options.ks, name=name+'_conv1')
#     h_db1 = denseblock(h_conv1, options, name=name+'_db1')    
#     h_maxpool1 = maxpool2d(h_db1, name=name+'_pool1')
#     h_db2 = denseblock(h_maxpool1, options, name=name+'_db2')
    
#     pooled_size = int(options.image_size / divide)
    
#     h_flat = tf.reshape(h_db2, [-1, pooled_size * pooled_size * options.nk])
#     h_fc1 = fully_connected(h_flat, options.nk * options.nk, name=name+'_fc1')
#     h_fc2 = fully_connected(h_fc1, options.n_pred, name=name+'_fc2')
    
#     return h_fc2
  
    
# def denseblock(input_, options, reuse=False, name='denseblock'):
    
#     with tf.variable_scope(name):
#         h_conv1 = conv2d(input_, options.nk, ks=options.ks, name='h_conv1')
#         h_conv2 = conv2d(tf.concat((input_,h_conv1),axis=3), options.nk, ks=options.ks, name='h_conv2')
#         h_conv3 = conv2d(tf.concat((input_,h_conv1,h_conv2),axis=3), options.nk, ks=options.ks, name='h_conv3')
#         h_conv4 = conv2d(tf.concat((input_,h_conv1,h_conv2,h_conv3),axis=3), options.nk, ks=options.ks, name='h_conv4')
#         h_conv5 = conv2d(tf.concat((input_,h_conv1,h_conv2,h_conv3,h_conv4),axis=3), options.nk, ks=options.ks, name='h_conv5')
        
#         return h_conv5
     
    