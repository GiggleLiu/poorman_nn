'''
Build a simple neural network that can be used to study mnist data set.
'''

import numpy as np
from poornn.nets import ANN
from poornn.checks import check_numdiff
from poornn import functions, Linear
from poornn.utils import typed_randn


def build_ann():
    '''
    builds a single layer network for mnist classification problem.
    '''
    F1 = 10
    I1, I2 = 28, 28
    eta = 0.1
    dtype = 'float32'

    W_fc1 = typed_randn(dtype, (F1, I1 * I2)) * eta
    b_fc1 = typed_randn(dtype, (F1,)) * eta

    # create an empty vertical network.
    ann = ANN()
    linear1 = Linear((-1, I1 * I2), dtype, W_fc1, b_fc1)
    ann.layers.append(linear1)
    ann.add_layer(functions.SoftMaxCrossEntropy, axis=1)
    ann.add_layer(functions.Mean, axis=0)
    return ann


# build and print it
ann = build_ann()
print(ann)

# random numerical differenciation check.
# prepair a one-hot target.
y_true = np.zeros(10, dtype='float32')
y_true[3] = 1

assert(all(check_numdiff(ann, var_dict={'y_true': y_true})))

# graphviz support
from poornn.visualize import viznn
viznn(ann, filename='./mnist_simple.png')
