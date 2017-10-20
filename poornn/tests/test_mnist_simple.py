from numpy import *
from numpy.testing import dec, assert_, assert_raises,\
    assert_almost_equal, assert_allclose
import sys
import pdb
import time
from copy import deepcopy
import argparse

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from climin import RmsProp, GradientDescent, Adam

from ..nets import ANN
from ..checks import check_numdiff
from .. import functions
from ..utils import typed_randn
from ..spconv import SPConv
from ..linears import Linear

FLAGS = None
data_cache = {}


def build_dnn():
    '''deepnn builds the graph for a deep net for classifying digits.'''
    # first convolution layer
    F1 = 10
    I1, I2 = 28, 28
    eta = 0.1
    dtype = 'float32'

    W_fc1 = typed_randn(dtype, (F1, I1 * I2)) * eta
    b_fc1 = typed_randn(dtype, (F1,)) * eta

    traditional_mode = False
    if traditional_mode:
        linear1 = Linear((-1, I1 * I2), dtype, W_fc1, b_fc1)
        # relu1 = functions.ReLU(linear1.output_shape, dtype)

        # the cost function
        # softmax = functions.SoftMax((-1,F1), dtype, axis=1)
        # costfunc2 = functions.CrossEntropy((-1,F1), dtype, axis=1)
        costfunc = functions.SoftMaxCrossEntropy((-1, F1), dtype, axis=1)

        meanfunc = functions.Mean((-1,), dtype, axis=0)
        ann = ANN(layers=[linear1, costfunc, meanfunc])
    else:
        ann = ANN()  # do not specify layers.
        linear1 = Linear((-1, I1 * I2), dtype, W_fc1, b_fc1)
        ann.layers.append(linear1)
        #ann.add_layer(functions.SoftMaxCrossEntropy, axis=1)
        ann.add_layer(functions.Normalize, axis=1)
        ann.add_layer(functions.SquareLoss, axis=1)
        ann.add_layer(functions.Sum, axis=1)
        ann.add_layer(functions.Mean, axis=0)

    # random num-diff check
    y_true = zeros(10, dtype='float32')
    y_true[3] = 1
    assert(all(check_numdiff(ann, var_dict={'y_true': y_true},tol=1e-2)))
    return ann


def compute_gradient(weight_vec, info_dict):
    dnn = info_dict['dnn']
    dnn.set_variables(weight_vec)
    dnn.set_runtime_vars({'y_true': info_dict['y_true']})
    x = info_dict['x_batch']
    y = dnn.forward(x, do_shape_check=True, data_cache=data_cache)
    gradient_w, gradient_x = dnn.backward((x, y), dy=ones_like(
        y), do_shape_check=True, data_cache=data_cache)
    info_dict['ys'] = data_cache['%s-ys' % id(dnn)]
    vec = gradient_w
    return vec


def analyse_result(ys, y_true):
    y_predict = ys[-4]
    correct_prediction = argmax(y_predict, axis=1) == argmax(y_true, axis=1)
    accuracy = mean(correct_prediction)
    print('Accuracy = %g' % (accuracy))
    return accuracy


def main(_):
    # Import data
    random.seed(2)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    dnn = build_dnn()
    var_vec = dnn.get_variables()
    info_dict = {'dnn': dnn}

    batch = mnist.train.next_batch(100)
    # add feature axis.
    info_dict['x_batch'] = batch[0].reshape([batch[0].shape[0], -1], order='F')
    info_dict['y_true'] = asfortranarray(batch[1], dtype=batch[0].dtype)
    optimizer = GradientDescent(wrt=var_vec,
                                fprime=lambda x: compute_gradient(
                                    x, info_dict), step_rate=0.5, momentum=0.)
    # optimizer=RmsProp(wrt=var_vec,
    # fprime=lambda x: compute_gradient(x,info_dict),
    # step_rate=1e-3,decay=0.9,momentum=0.9)

    for k, info in enumerate(optimizer):
        if k % 100 == 0:
            print('Analyse Step = %s' % k)
            print('Cost = %s' % info_dict['ys'][-1])
            analyse_result(info_dict['ys'], info_dict['y_true'])
        batch = mnist.train.next_batch(100)
        # add feature axis.
        info_dict['x_batch'] = asfortranarray(batch[0],
                                              dtype='float32').reshape([
                                                  batch[0].shape[0], -1],
                                                  order='F')
        info_dict['y_true'] = asfortranarray(batch[1], dtype='float32')
        if k > 5000:
            break

    # apply on test cases
    dnn.set_runtime_vars({'y_true': mnist.test.labels})
    ys = dnn.forward(mnist.test.images, data_cache=data_cache)
    analyse_result(data_cache['%d-ys' % id(dnn)], mnist.test.labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='MNIST-data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
