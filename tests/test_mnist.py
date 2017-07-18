from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time
import argparse
sys.path.insert(0,'../')

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from climin import RmsProp,GradientDescent,Adam
from core import ANN
import functions
from spconv import SPConv
from linears import Linear
from utils import unpack_variables, pack_variables

FLAGS = None

randn=lambda *args,**kwargs:random.randn(*args,**kwargs)*0.1

def build_dnn():
    '''deepnn builds the graph for a deep net for classifying digits.'''
    #first convolution layer
    K1=5
    F1=32
    I1, I2 = 28, 28
    img_in_shape=(I1,I2)
    W_conv1 = randn(F1, 1, K1, K1)  #fout, fin, K1, K2
    b_conv1 = randn(F1)
    conv1=SPConv((-1,1)+img_in_shape, W_conv1, bias=b_conv1, dtype='float32', strides=(1,1), boundary='P')
    relu1 = functions.ReLU()
    pooling1 = functions.MaxPool(input_shape=conv1.output_shape, kernel_shape=(2,2), boundary='O')

    #second convolution layer
    F2=64
    W_conv2 = randn(F2, F1, K1, K1)  #fout, fin, K1, K2
    b_conv2 = randn(F2)
    conv2=SPConv(pooling1.output_shape, W_conv2, bias=b_conv2, dtype='float32', strides=(1,1), boundary='P')
    relu2 = functions.ReLU()
    pooling2 = functions.MaxPool(kernel_shape=(2,2), input_shape=conv2.output_shape, output_shape=(-1,F2*I1/4*I2/4), boundary='O')

    #fully connected layer
    F3=1024
    W_fc1 = randn(F3, pooling2.output_shape[1])
    b_fc1 = randn(F3)
    linear1 = Linear(W_fc1, b_fc1)
    dropout1=functions.DropOut(input_shape=(-1,F3),keep_rate=0.5, axis=1)

    F4=10
    W_fc2 = randn(F4, F3)
    b_fc2 = randn(F4)
    linear2 = Linear(W_fc2, b_fc2)

    #the cost function
    costfunc = functions.SoftMaxCrossEntropy(input_shape=(-1,F4),axis=1)
    meanfunc = functions.Mean((-1,),axis=0)
    return ANN([conv1, relu1, pooling1, conv2, relu2, pooling2, linear1, dropout1, linear2, costfunc, meanfunc])

def compute_gradient(weight_vec, info_dict):
    dnn=info_dict['dnn']
    dnn.set_variables(unpack_variables(weight_vec, info_dict['shapes']))
    ys = dnn.feed_input(info_dict['x_batch'], info_dict['y_true'])
    gradient_w, gradient_x = dnn.back_propagate(ys, dy=ones_like(ys[-1]))
    info_dict['gradient_x'] = gradient_x
    info_dict['ys'] = ys
    vec, shapes = pack_variables(gradient_w)
    info_dict['shapes'] = shapes
    return vec

def analyse_result(ys, y_true):
    y_predict = ys[-3]
    correct_prediction = argmax(y_predict, axis=1)==argmax(y_true, axis=1)
    accuracy = mean(correct_prediction)
    print('training accuracy = %g' % (accuracy))
    return accuracy

def main(_):
    # Import data
    random.seed(2)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    dnn = build_dnn()
    var_vec, shapes = pack_variables(dnn.get_variables())
    info_dict = {'dnn':dnn, 'shapes':shapes}

    batch = mnist.train.next_batch(50)
    info_dict['x_batch'] = batch[0].reshape([batch[0].shape[0],1,28,28],order='F')   #add feature axis.
    info_dict['y_true'] = asfortranarray(batch[1],dtype=batch[0].dtype)
    optimizer=RmsProp(wrt=var_vec,fprime=lambda x: compute_gradient(x,info_dict),step_rate=1e-3,decay=0.9,momentum=0.)

    for k,info in enumerate(optimizer):
        batch = mnist.train.next_batch(50)
        info_dict['x_batch'] = batch[0].reshape([batch[0].shape[0],1,28,28],order='F')   #add feature axis.
        info_dict['y_true'] = batch[1].reshape([-1,10],order='F')
        if k % 100 == 0:
            print 'Analyse Step = %s'%k
            analyse_result(info_dict['ys'], info_dict['y_true'])
        if k>10000: break

    #apply on test cases
    ys = dnn.forward(mnist.test.images)
    analyse_result(ys, mnist.test.labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/home/leo/bigdata/MNIST-data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
