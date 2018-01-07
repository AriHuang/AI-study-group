import tensorflow as tf

def weight_variable(shape, trainable = True):
    initializer = tf.truncated_normal_initializer(0.0, stddev=0.05)
    return tf.get_variable(name = 'weights', shape = shape, initializer=initializer, trainable=trainable)

def bias_variable(shape, trainable = True):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable(name = 'bias', shape = shape, initializer=initializer, trainable=trainable)

def conv2d(input, in_features, out_features, kernel_size, with_bias=False, relu = True, name = None):
    with tf.variable_scope(name) as scope:
        W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
        conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
        print(W)
        if with_bias:
            conv = conv + bias_variable([ out_features ])
        if relu:
            conv = tf.nn.relu(conv)
        return conv

def deconv2d(input, in_features, out_features, kernel_size, strides =1, with_bias=False, relu = True, name = None):
    in_shape = tf.shape(input)
    output_shape = [in_shape[0], in_shape[1]*strides, in_shape[2]*strides,  out_features]
    with tf.variable_scope(name) as scope:
        W = weight_variable([ kernel_size, kernel_size, out_features, in_features ])
        deconvolve = tf.nn.conv2d_transpose(input, W, output_shape, strides = [ 1, strides, strides, 1 ], padding='SAME')
        if with_bias:
            deconvolve = deconvolve + bias_variable([ out_features ])
        if relu:
            deconvolve = tf.nn.relu(deconvolve)
        return deconvolve


def avg_pool(input, stride = 2):
    return tf.nn.avg_pool(input, [ 1, stride, stride, 1 ], [1, stride, stride, 1 ], 'VALID')

def max_pool(input, stride =2 ):
    return tf.nn.max_pool(input, [ 1, stride, stride, 1 ], [1, stride, stride, 1 ], 'VALID')


def restoreState(session , save_path, collection = None, with_bias = True):
    if collection == None:
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(session, checkpoint.model_checkpoint_path)
            print( "Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print( "Could not find old network weights")
    else:
        keys = []
        for name in collection:
            with tf.variable_scope(name, reuse=True):
                keys.append(tf.get_variable('weights'))
                if with_bias:
                    keys.append(tf.get_variable('biases'))
        saver = tf.train.Saver(keys)
        print("restore collection:,", collection, keys)
        checkpoint = tf.train.get_checkpoint_state(save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(session, checkpoint.model_checkpoint_path)
            print( "Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print( "Could not find old network weights")