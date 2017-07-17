from numpy.random import randn
from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time
sys.path.insert(0,'../')

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import functions
from spconv import SPConv
from linears import Linear

FLAGS = None

def build_dnn():
    '''deepnn builds the graph for a deep net for classifying digits.'''
    #first convolution layer
    K1=5
    F1=32
    img_in_shape=(28,28)
    W_conv1 = randn([F1, 1, K1, K1])  #fout, fin, K1, K2
    b_conv1 = randn([F1])
    conv1=SPConv(W_conv1, bias=b_conv1, img_in_shape=img_in_shape, dtype='float32', strides=(1,1), boundary='P')
    relu1 = functions.ReLU()
    pooling1 = functions.MaxPool(kernel_shape=(2,2), img_in_shape=conv1.img_out_shape, boundary='O')

    #second convolution layer
    F2=64
    W_conv2 = randn([F2, 1, K1, K1])  #fout, fin, K1, K2
    b_conv2 = randn([F2])
    conv2=SPConv(W_conv2, bias=b_conv2, img_in_shape=pooling1.img_out_shape, dtype='float32', strides=(1,1), boundary='P')
    relu2=relu1
    pooling2 = functions.MaxPool(kernel_shape=(2,2), img_in_shape=conv2.img_out_shape, boundary='O')

    #fully connected layer
    F3=1024
    W_fc1 = randn([F3, np.prod(pooling2.img_out_shape)*F2])
    b_fc1 = randn([F3])
    linear1 = Linear(W_fc1, b_fc1)
    dropout1=functions.DropOut(0.5)

    F4=10
    W_fc2 = randn([F4, F3])
    b_fc2 = randn([F4])
    linear2 = Linear(W_fc2, b_fc2)

    #the cost function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    return ANN([conv1, relu1, pooling1, conv2, relu2, pooling2, linear1, dropout1, linear2])

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    dnn=build_dnn()

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  #return an operation
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/home/leo/bigdata/MNIST-data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
