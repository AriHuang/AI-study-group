##
## author: sting.huang <arihuang@hotmail.com>
##

import tensorflow as tf
import os


DEFAULT_PADDING = 'SAME'
LEAKY_RELU = False

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name',  self.makename())
        if len(self.feedlayer)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        layer_input = self.feedlayer[-1] # the input is always the last output
        layer_output = op( self, layer_input, *args, **kwargs)
        self.feedlayer.append(layer_output)
        return self
    return layer_decorated

class network:
    def __init__(self, trainable = True , savepath = "save", autoencoder = False, name = None):
        assert type(trainable) != str
        self.variable_list = []
        self.feedlayer = []
        self.statemap = []
        self.trainable = trainable
        self.padding = DEFAULT_PADDING
        self.namecunt = 0
        self.setup( autoencoder, name)
        self.save_path = savepath
        if os.path.isdir(savepath) == False:
            os.makedirs(savepath)


    def setup(self, autoencoder, name = None):
        raise NotImplementedError('Must be subclassed.')

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    #   collection, assgin the various need to save and load
    def loadsavedstate(self, session , collection = None , save_file = None):
        if collection == None:
            self.saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(self.save_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(session, checkpoint.model_checkpoint_path)
                print( "Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print( "Could not find old network weights")
        else:
            keys = []
            for name in collection:
                with tf.variable_scope(name, reuse=True):
                    keys.append(tf.get_variable('weights'))
                    keys.append(tf.get_variable('biases'))
            self.saver = tf.train.Saver(keys)
            print("restore collection:,", collection, keys)
            if save_file == None:
                checkpoint = tf.train.get_checkpoint_state(self.save_path)
                if checkpoint and checkpoint.model_checkpoint_path:
                    self.saver.restore(session, checkpoint.model_checkpoint_path)
                    print( "Successfully loaded:", checkpoint.model_checkpoint_path)
                else:
                    print( "Could not find old network weights")
            else:
                self.saver.restore(session, save_file)
                print( "Successfully loaded:", save_file)
            self.saver = tf.train.Saver()

    def makename(self):
        self.namecunt += 1
        return "unamed_" + str(self.namecunt)

    def feed(self, inputs):
        #assert len(inputs)!=0
        self.feedlayer = []
        self.feedlayer.append(inputs)
        return self

    def get_ouput(self):
        assert len(self.feedlayer)!=0
        return  self.feedlayer[-1]

    #init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
    #init_biases = tf.constant_initializer(0.0)
    def make_variable(self, name, shape, initializer = None, trainable = True):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
        self.variable_list.append(var)
        return var

    def copynetwork( self, sess, net):
        qnet = net.variable_list[:]
        op = []
        for n in range(len(self.variable_list)):
            op.append(self.variable_list[n].assign(qnet[n]))
            #sess.run( self.variable_list[n].assign(qnet[n]))
        sess.run(op)

        #Given an input tensor of shape [batch, in_height, in_width, in_channels] and
    #a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #strides = [1, stride_height, stride_width, 1]
    @layer
    def conv2d( self, inputs, mask, out_channel,   strides = [1,1] , relu=True, padding = None, trainable=True, name = None):
        padding = self.padding if padding == None else padding
        self.validate_padding(padding)
        in_channel = inputs.get_shape()[-1]
        assert  in_channel != 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, strides[0], strides[1], 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.05)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_variable('weights', [mask[0], mask[1], in_channel, out_channel], init_weights, trainable)
            biases = self.make_variable('biases', out_channel, init_biases, trainable)
            conv =  convolve(inputs, kernel)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return self.__relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def deconv2d( self, inputs, batch_size, out_channel,   strides = [1,1] , relu=True, padding = None, trainable=True, name = None):
        padding = self.padding if padding == None else padding
        self.validate_padding(padding)
        in_shape =  inputs.get_shape().as_list()
        assert   len(in_shape) == 4
        output_shape = [batch_size, in_shape[1]*strides[0], in_shape[2]*strides[1],  out_channel]
        deconvolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape, strides = [1, strides[0], strides[1], 1], padding=padding)
        with tf.variable_scope(name, reuse=True) as scope:
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')
            conv =  deconvolve(inputs, kernel)
            if relu:
                #bias = tf.nn.bias_add(conv, biases)
                return self.__relu(conv, name=scope.name)
            return conv

    @layer
    def deconv2du( self, inputs, mask,  batch_size, out_channel,   strides = [1,1] , relu=True, padding = None, trainable=True, name = None):
        padding = self.padding if padding == None else padding
        self.validate_padding(padding)
        in_shape =  inputs.get_shape().as_list()
        in_channel = inputs.get_shape()[-1]
        assert   len(in_shape) == 4
        output_shape = [batch_size, in_shape[1]*strides[0], in_shape[2]*strides[1],  out_channel]
        deconvolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape, strides = [1, strides[0], strides[1], 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_variable('weights', [mask[0], mask[1], out_channel, in_channel], init_weights, trainable)
            biases = self.make_variable('biases', in_channel, init_biases, trainable)
            bias = tf.nn.bias_add(inputs, biases)
            conv =  deconvolve(bias, kernel)
            if relu:
                return self.__relu(conv, name=scope.name)
            return conv

    @layer
    def max_pool(self,  inputs, mask = [2, 2] , batch = 1, ch = 1, strides = [2,2], padding=None,  name = None):
        padding = self.padding if padding == None else padding
        self.validate_padding(padding)
        return tf.nn.max_pool(inputs,
                              ksize=[batch, mask[0], mask[1], ch],
                              strides=[1, strides[0], strides[1], 1],
                              padding=padding,
                              name=name)

    @layer
    def ave_pool(self,  inputs, mask = [2, 2] , batch = 1, ch = 1, strides = [2,2], padding=None,  name = None):
        padding = self.padding if padding == None else padding
        self.validate_padding(padding)
        return tf.nn.avg_pool(inputs,
                              ksize=[batch, mask[0], mask[1], ch],
                              strides=[1, strides[0], strides[1], 1],
                              padding=padding,
                              name=name)

    @layer
    def relu(self, inputs, name = None):
        return self.__relu(inputs, name)

    def __relu(self, inputs, name = None):
        if LEAKY_RELU == True:
            leakiness = 0.1
            return tf.where(tf.less(inputs, 0.0), leakiness * inputs, inputs, name=name)
        return tf.nn.relu(inputs, name=name)

    @layer
    def tanh(self, inputs, name = None):
        return tf.nn.tanh(inputs, name=name)

    @layer
    def sigmoid(self, inputs, name = None):
        return tf.sigmoid(inputs, name=name)


    @layer
    def fc(self, inputs, num_out, relu=True, trainable=True, name =None):
        with tf.variable_scope(name) as scope:
            # only use the first input
            #if isinstance(input, tuple):
            #    input = input[0]

            input_shape = inputs.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                #feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
                feed_in = tf.reshape(inputs, [-1, dim])
            else:
                feed_in, dim = (inputs, int(input_shape[-1]))

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_variable('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_variable('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else  tf.nn.xw_plus_b # same as ==> lambda x, w, b: tf.matmul(x, w) + b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, inputs, name = None):
        input_shape = tf.shape(inputs)
        return tf.nn.softmax(inputs,name=name)

    @layer
    def dropout(self, inputs, keep_prob = 1 , name= None):
        return tf.nn.dropout(inputs, keep_prob, name=name)

    @layer
    def batch_norm(self, inputs, phase , name = None):
        return tf.contrib.layers.batch_norm(inputs,
                                            center=True, scale=True,
                                            is_training=phase,
                                            scope= name)
    @layer
    def makeflat(self, inputs, name = None):
        input_shape = inputs.get_shape()
        dim = 1
        for d in input_shape[1:].as_list():
            dim *= d
        return   tf.reshape(inputs, [-1, dim])


    def _global_ave_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    @layer
    def global_ave_pool(self,  inputs, name = None):
        return self._global_ave_pool(inputs)






