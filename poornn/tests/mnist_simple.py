from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time
from copy import deepcopy
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
randn=lambda *args,**kwargs:random.normal(*args,**kwargs)*0.001

def build_dnn():
    '''deepnn builds the graph for a deep net for classifying digits.'''
    #first convolution layer
    F1=10
    I1, I2 = 28, 28

    W_fc1 = randn(size=(F1, I1*I2))
    b_fc1 = randn(size=(F1))
    linear1 = Linear(W_fc1, b_fc1, dtype='float32')
    relu1 = functions.ReLU_I()

    #the cost function
    softmax = functions.SoftMax(input_shape=(-1,F1),axis=1)
    costfunc2 = functions.CrossEntropy(input_shape=(-1,F1),axis=1)
    costfunc = functions.SoftMaxCrossEntropy(input_shape=(-1,F1),axis=1)

    meanfunc = functions.Mean(input_shape=(-1,),axis=0)
    return ANN([linear1, costfunc, meanfunc])
    #return ANN([linear1, softmax, costfunc2, meanfunc])

def compute_gradient(weight_vec, info_dict):
    dnn=info_dict['dnn']
    dnn.set_variables_vec(weight_vec)
    ys = dnn.feed_input(info_dict['x_batch'], info_dict['y_true'])
    gradient_w, gradient_x = dnn.back_propagate(ys, dy=ones_like(ys[-1]))
    info_dict['ys'] = ys
    vec, shapes = pack_variables(gradient_w)
    return vec

def analyse_result(ys, y_true):
    y_predict = ys[-3]
    correct_prediction = argmax(y_predict, axis=1)==argmax(y_true, axis=1)
    accuracy = mean(correct_prediction)
    print('Accuracy = %g' % (accuracy))
    return accuracy

def main(_):
    # Import data
    random.seed(2)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    dnn = build_dnn()
    var_vec = dnn.get_variables_vec()
    info_dict = {'dnn':dnn}

    batch = mnist.train.next_batch(100)
    info_dict['x_batch'] = batch[0].reshape([batch[0].shape[0],-1],order='F')   #add feature axis.
    info_dict['y_true'] = asfortranarray(batch[1],dtype=batch[0].dtype)
    optimizer=GradientDescent(wrt=var_vec,fprime=lambda x: compute_gradient(x,info_dict),step_rate=0.5,momentum=0.)
    #optimizer=RmsProp(wrt=var_vec,fprime=lambda x: compute_gradient(x,info_dict),step_rate=1e-3,decay=0.9,momentum=0.9)

    for k,info in enumerate(optimizer):
        if k % 100 == 0:
            print 'Analyse Step = %s'%k
            print 'Cost = %s'%info_dict['ys'][-1]
            analyse_result(info_dict['ys'], info_dict['y_true'])
        batch = mnist.train.next_batch(100)
        info_dict['x_batch'] = asfortranarray(batch[0], dtype='float32').reshape([batch[0].shape[0],-1],order='F')   #add feature axis.
        info_dict['y_true'] = asfortranarray(batch[1],dtype='float32')
        if k>5000: break

    #apply on test cases
    ys = dnn.feed_input(mnist.test.images, y_true=mnist.test.labels)
    analyse_result(ys, mnist.test.labels)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/home/leo/bigdata/MNIST-data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
