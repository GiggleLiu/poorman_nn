import numpy as np
import pdb

from utils import typed_randn


__all__=['dec_check_shape', 'check_numdiff', 'generate_randx',
        'check_shape_backward', 'check_shape_forward', 'check_shape_match']

def dec_check_shape(pos):
    '''
    Check the shape of layer's method.
    Return a decorator.
    '''
    def real_decorator(f):
        def wrapper(*args, **kwargs):
            #x, y, dy
            argss=list(args)
            if len(argss)==1:
                argss.append(kwargs.get('x'))
            if len(argss)==2:
                argss.append(kwargs.get('y'))
            if len(argss)==3:
                argss.append(kwargs.get('dy'))
            layer = args[0]
            for p in pos:
                if p > 0:
                    _check_input(layer, argss[p])
                else:
                    _check_output(layer, argss[-p])
            res = f(*args, **kwargs)
            if isinstance(res, tuple):  #backward, dw, dx
                _check_input(layer, res[1])
            else:
                _check_output(layer, res)
            return res
        return wrapper
    return real_decorator

def generate_randx(layer):
    '''Generate random input tensor.'''
    max_dim=3
    max_size=20
    input_shape = layer.input_shape
    dtype = layer.dtype
    if input_shape is None:
        return typed_randn(dtype, np.random.randint(1, max_size+1, np.random.randint(1,max_dim+1)))
    input_shape = [n if n>=0 else np.random.randint(1, max_size+1) for n in input_shape]
    return typed_randn(dtype, input_shape)

def check_shape_forward(f):
    '''
    Check the shape of layer's method.
    '''
    def wrapper(*args, **kwargs):
        #x, y, dy
        layer = args[0]
        x=kwargs.get('x') if len(args)==1 else args[1]
        _check_input(layer, x)
        res = f(*args[1:], **kwargs)
        _check_output(layer, res)
        return res
    return wrapper

def check_shape_backward(f):
    '''
    Check the shape of layer's method.
    '''
    def wrapper(*args, **kwargs):
        #x, y, dy
        argss=list(args)
        if len(argss)==1:
            argss.append(kwargs.get('xy'))
        if len(argss)==2:
            argss.append(kwargs.get('dy'))
        layer = args[0]
        _check_input(layer, argss[1][0])
        _check_output(layer, argss[1][1])
        _check_output(layer, argss[2])
        res = f(*args[1:], **kwargs)
        _check_input(layer, res[1])
        return res
    return wrapper

def check_numdiff(layer, x=None, num_check=10, eta=None, tol=1e-1, var_dict={}):
    '''Random Numerical Differential check.'''
    if x is None:
        x=generate_randx(layer)
    else:
        x=np.asfortranarray(x, dtype=layer.dtype)
    is_net = hasattr(layer, 'num_layers')

    layer.set_runtime_vars(var_dict)
    ys=layer.forward(x)
    if is_net:
        y=ys[-1]
        dv, dx=layer.backward(ys, dy=np.ones_like(y))
    else:
        y=ys
        dv, dx=layer.backward([x,y], dy=np.ones_like(y))
    dx_=np.ravel(dx, order='F')
    if eta is None:
        eta=0.003+0.004j if np.iscomplexobj(dv) else 0.005

    res_x=[]
    #check dy/dx
    for i in range(num_check):
        #change variables at random position.
        pos=np.random.randint(0,x.size)
        x_new=x.copy(order='F')
        x_new.ravel(order='F')[pos]+=eta
        y1=layer.forward(x_new)
        if is_net: y1=y1[-1]

        diff=abs(dx_[pos]-np.sum(y1-y)/eta)
        if diff/max(1,abs(dx_[pos]))>tol:
            print 'XBP Diff = %s, Num Diff = %s'%(dx_[pos], np.sum(y1-y)/eta)
            print 'Num Diff Test Fail! @x_[%s] = %s'%(pos, x.ravel()[pos])
            res_x.append(False)
        else:
            res_x.append(True)

    if not is_net and layer.num_variables==0:
        return res_x

    #check dy/dw
    res_w=[]
    var0 = layer.get_variables()
    for i in range(num_check):
        #change variables at random position.
        var=var0.copy()
        pos=np.random.randint(0,var.size)
        var[pos]+=eta
        layer.set_variables(var)
        y1=layer.forward(x)
        if is_net: y1=y1[-1]

        diff=abs(dv[pos]-np.sum(y1-y)/eta)
        if diff/max(1, abs(dv[pos]))>tol:
            print 'WBP Diff = %s, Num Diff = %s'%(dv[pos], np.sum(y1-y)/eta)
            print 'Num Diff Test Fail! @var[%s] = %s'%(pos,var[pos])
            res_w.append(False)
        else:
            res_w.append(True)
    return res_w+res_x

def generate_randx(layer):
    '''Generate random input tensor.'''
    max_dim=3
    max_size=20
    input_shape = layer.input_shape
    dtype = layer.dtype
    if input_shape is None:
        return typed_randn(dtype, np.random.randint(1, max_size+1, np.random.randint(1,max_dim+1)))
    input_shape = [n if n>=0 else np.random.randint(1, max_size+1) for n in input_shape]
    return typed_randn(dtype, input_shape)

def _check_input(layer, x):
    if layer.input_shape is None or x is None:
        return
    if x.ndim!=len(layer.input_shape):
        raise ValueError('Dimension mismatch! x %s, desire %s'%(x.ndim, len(layer.input_shape)))
    for shape_i, xshape_i in zip(layer.input_shape, x.shape):
        if shape_i!=-1 and shape_i!=xshape_i:
            raise ValueError('Illegal Input shape! x %s, desire %s'%(x.shape, layer.input_shape))

def _check_output(layer, y):
    if layer.output_shape is None or y is None:
        return
    if y.ndim!=len(layer.output_shape):
        raise ValueError('Dimension mismatch! y %s, desire %s'%(y.ndim, len(layer.output_shape)))
    for shape_i, yshape_i in zip(layer.output_shape, y.shape):
        if shape_i!=-1 and shape_i!=yshape_i:
            raise ValueError('Illegal Output shape! y %s, desire %s'%(y.shape, layer.output_shape))


def check_shape_match(shape_get, shape_desire):
    '''
    check whether shape_get matches shape_desire.

    Parameters:
        :shape_get: tuple, obtained shape.
        :shape_desire: tuple, desired shape.

    Return:
        tuple, the shape with more details.
    '''
    #match None shapes
    if shape_get is None:
        return shape_desire
    if shape_desire is None:
        return shape_get

    #dimension mismatch
    if shape_get!=len(shape_desire):
        raise ValueError('Dimension mismatch! y %s, desire %s'%(y.ndim, len(layer.output_shape)))

    #element wise check
    shape=[]
    for shape_get_i, shape_desire_i in zip(shape_get, shape_desire):
        if shape_desire_i==-1:
            shape.append(shape_get_i)
        elif shape_get_i==-1:
            shape.append(shape_desire_i)
        else:
            if shape_desire_i!=shape_get_i:
                raise ValueError('Shape Mismatch! get %s, desire %s'%(shape_get, shape_desire))
            else:
                shape.append(shape_get_i)
    return tuple(shape)
