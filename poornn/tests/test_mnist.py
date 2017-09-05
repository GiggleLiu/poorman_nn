from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time,os
import argparse

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from climin import RmsProp,GradientDescent,Adam

from ..nets import ANN
from .. import functions
from ..spconv import SPConv
from ..linears import Linear
from ..checks import check_numdiff
from ..utils import typed_randn
from ..visualize import viznn

random.seed(2)
FLAGS = None

def build_dnn():
    '''deepnn builds the graph for a deep net for classifying digits.'''
    #first convolution layer
    K1=5
    F1, F2, F3, F4 = 32, 64, 1024, 10
    F3=1024
    I1, I2 = 28, 28
    dtype = 'float32'
    eta=0.1

    ann=ANN(itype=dtype, do_shape_check=True)

    W_conv1 = eta*typed_randn(dtype, (F1, 1, K1, K1))  #fout, fin, K1, K2
    b_conv1 = eta*typed_randn(dtype, (F1,))
    ann.layers.append(SPConv(input_shape=(-1,1,I1,I2), itype=dtype, weight=W_conv1, bias=b_conv1, strides=(1,1), boundary='P'))
    ann.add_layer(functions.ReLU)
    ann.add_layer(functions.Pooling, mode='max', kernel_shape=(2,2), boundary='O')

    #second convolution layer
    W_conv2 = eta*typed_randn(dtype, (F2, F1, K1, K1))  #fout, fin, K1, K2
    b_conv2 = eta*typed_randn(dtype, (F2,))
    ann.add_layer(SPConv, weight=W_conv2, bias=b_conv2, strides=(1,1), boundary='P')
    ann.add_layer(functions.ReLU)
    ann.add_layer(functions.Pooling, mode='max', kernel_shape=(2,2), boundary='O')

    nout=prod(ann.layers[-1].output_shape[1:])
    ann.add_layer(functions.Reshape, output_shape=(-1, nout))

    #fully connected layer
    W_fc1 = eta*typed_randn(dtype, (F3, nout))
    b_fc1 = eta*typed_randn(dtype, (F3,))
    ann.add_layer(Linear, weight=W_fc1, bias=b_fc1)
    ann.add_layer(functions.ReLU)
    ann.add_layer(functions.DropOut, keep_rate=0.5, axis=1)

    W_fc2 = eta*typed_randn(dtype, (F4, F3))
    b_fc2 = eta*typed_randn(dtype, (F4,))
    ann.add_layer(Linear, weight=W_fc2, bias=b_fc2)

    #the cost function
    ann.add_layer(functions.SoftMaxCrossEntropy, axis=1)
    ann.add_layer(functions.Mean, axis=0)

    #random num-diff check
    y_true=zeros(F4); y_true[3]=1
    assert(all(check_numdiff(ann, var_dict={'y_true':y_true, 'seed':2}, eta=1e-3)))
    viznn(ann, filename='%s/data/test_mnist.pdf'%os.path.dirname(__file__))
    return ann

def compute_gradient(weight_vec, info_dict):
    dnn=info_dict['dnn']
    dnn.set_variables(weight_vec)
    dnn.set_runtime_vars({'y_true':info_dict['y_true'], 'seed':random.randint(1,99999)})
    ys = dnn.forward(info_dict['x_batch'])
    gradient_w, gradient_x = dnn.backward(ys, dy=ones_like(ys[-1]))
    info_dict['gradient_x'] = gradient_x
    info_dict['ys'] = ys
    vec = gradient_w
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
    var_vec = dnn.get_variables()
    info_dict = {'dnn':dnn}

    batch = mnist.train.next_batch(50)
    #info_dict['x_batch'] = asfortranarray(batch[0]).reshape([batch[0].shape[0],-1],order='F')   #add feature axis.
    info_dict['x_batch'] = asfortranarray(batch[0]).reshape([batch[0].shape[0],1,28,28],order='F')   #add feature axis.
    info_dict['y_true'] = asfortranarray(batch[1],dtype=batch[0].dtype)
    optimizer=RmsProp(wrt=var_vec,fprime=lambda x: compute_gradient(x,info_dict),step_rate=1e-3,decay=0.9,momentum=0.)
    #optimizer=GradientDescent(wrt=var_vec,fprime=lambda x: compute_gradient(x,info_dict),step_rate=1e-2,momentum=0.)
    #optimizer=Adam(wrt=var_vec,fprime=lambda x: compute_gradient(x,info_dict),step_rate=1e-3)

    t0=time.time()
    for k,info in enumerate(optimizer):
        if k % 10 == 0:
            t1=time.time()
            print(t1-t0)
            t0=time.time()
            print('Analyse Step = %s'%k)
            analyse_result(info_dict['ys'], info_dict['y_true'])
        batch = mnist.train.next_batch(50)
        #info_dict['x_batch'] = asfortranarray(batch[0]).reshape([batch[0].shape[0],-1],order='F')   #add feature axis.
        info_dict['x_batch'] = asfortranarray(batch[0]).reshape([batch[0].shape[0],1,28,28],order='F')   #add feature axis.
        info_dict['y_true'] = asarray(batch[1],order='F',dtype='float32')
        if k>8000: break

    #apply on test cases
    dnn.layers[7].keep_rate=1.
    dnn.set_runtime_vars({'y_true':mnist.test.labels, 'seed':random.randint(1,99999)})
    ys = dnn.forward(asfortranarray(mnist.test.images).reshape([-1,1,28,28], order='F'))
    #ys = dnn.forward(mnist.test.images, mnist.test.labels)
    analyse_result(ys, mnist.test.labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='MNIST-data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
