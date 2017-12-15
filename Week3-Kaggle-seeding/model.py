
import numpy as np
import tensorflow as tf

from network import network
from seedingimage import dataset


LR = 1e-4
CLASS_NUM = 12
IMG_H = 128
IMG_W = 128

class Model(network):
    def setup(self, autoencoder, name = None):
        assert  name != None
        self.padding = "SAME"
        with tf.variable_scope(name):
            self.image = tf.placeholder(tf.float32, [None, IMG_H,IMG_W, 3], name="image")
            if self.trainable == True: # for test and validation
                self.keep_prob = tf.placeholder(tf.float32)
                self.y_ = tf.placeholder(tf.float32, [None, CLASS_NUM], name="labels")
            (self.feed(self.image )
                .conv2d(mask = [3,3], out_channel = 16, strides = [1,1], name = "l1con1")  #128*128*3 -> 128*128*16
                .conv2d(mask = [3,3], out_channel = 16, strides = [1,1], name = "l1con2")  #128*128*16 -> 128*128*16
                .max_pool(mask = [2,2])  #128*128*16 -> 64*64*16
                .conv2d(mask = [3,3], out_channel = 32, strides = [1,1], name = "l2con1")  #64*64*16 -> 64*64*32
                .conv2d(mask = [3,3], out_channel = 32, strides = [1,1], name = "l2con2")  #64*64*32 -> 64*64*32
                .max_pool(mask = [2,2])  #64*64*32 -> 32*32*32
                .conv2d(mask = [3,3], out_channel = 64, strides = [1,1], name = "l3con1")  #32*32*32 -> 32*32*64
                .makeflat() # 32*32864 -> 65536
                .dropout(keep_prob = self.keep_prob)
                .fc(512, relu = True, trainable = self.trainable, name = "fc1")
                .dropout(keep_prob = self.keep_prob)
                .fc(256, relu = True, trainable = self.trainable, name = "fc2")
                .dropout(keep_prob = self.keep_prob)
                .fc(CLASS_NUM, relu = False, trainable = self.trainable, name = "out")
            )

            self.y = self.get_ouput()
            print( self.y, np.shape(self.y_))
            self.predict_soft =  tf.nn.softmax(self.y, name = "soft")
            self.predict = tf.argmax(self.y, axis = 1, name = "predict")
            if self.trainable == True: # for test and validation
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels = self.y_, logits = self.y,  name='loss_'), name = "loss")
                correct_prediction = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.y, 1))
                self.accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.trainStep =  tf.train.AdamOptimizer(LR, name = "train_opt").minimize(self.cost)


    def dumpvarious(self, sess, filename):
        with open( filename,'w') as fname:
            for var in self.variable_list:
                fname.write(str(var))
                fname.write('\r')
                #fname.write(str(sess.run(var.read_value())))
                #print(var.read_value())

if __name__ == '__main__':
    train_db = dataset("train")
    valid_db = dataset("validation")
    net = Model(name= "CNN")
    KEEP_PROB = 0.5
    echo = 0
    patient = 10
    best_acc = 0
    best_valid_acc = 0
    best_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while(True):
            echo += 1
            x_, y_ = train_db.get_batch(32)
            feed_dict = {net.image: x_, net.y_: y_, net.keep_prob: KEEP_PROB}
            sess.run(net.trainStep, feed_dict = feed_dict)
            if echo % 50 == 0:
                x_, y_ = train_db.get_batch(64)
                feed_dict = {net.image: x_, net.y_: y_, net.keep_prob: 1}
                cost, acc = sess.run( (net.cost, net.accuracy), feed_dict = feed_dict)
                x_, y_ = valid_db.get_batch(120)
                feed_dict = {net.image: x_, net.y_: y_, net.keep_prob: 1}
                valid_cost, valid_acc = sess.run( (net.cost, net.accuracy), feed_dict = feed_dict)
                print("Step:{0}, training cost:{1:.3f}, accuracy:{2:.3f}, validation cost:{3:.3f}, accuracy:{4:.3f}".format( echo, cost, acc, valid_cost,valid_acc))
                patient -= 1
                if best_acc < acc:
                    best_acc = acc
                    best_valid_acc = valid_acc
                    patient = 10
                    best_step = echo
            if patient <= 0:
                print("Early stop on step {0} with accuracy:{1:.3f} and validation accuracy:{2:.3f}".format( best_step, best_acc, best_valid_acc))
                break