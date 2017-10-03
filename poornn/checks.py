import numpy as np
import pdb

from .utils import typed_randn


__all__=['dec_check_shape', 'check_numdiff', 'generate_randx',
        'check_shape_backward', 'check_shape_forward', 'check_shape_match']

# TODO: Error handling, non-differentiable error.

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

def check_numdiff(layer, x=None, num_check=10, eta_x=None, eta_w=None, tol=1e-5, var_dict={}):
    '''Random Numerical Differentiation check.'''
    from .nets import ANN
    analytical = layer.tags.get('analytical',1) if hasattr(layer,'tags') else 1
    if analytical == 0 or (analytical==3 and layer.itype[:7]=='complex'):
        print('Warning: Layer %s is not analytic, going on numdiff check!'%layer)
    is_net = isinstance(layer, ANN)

    # generate input and set runtime input
    input_dtype = layer.itype
    if x is None:
        x=generate_randx(layer)
    else:
        x=np.asarray(x, dtype=input_dtype, order='F')

    # forward to generate ys, and y.
    layer.set_runtime_vars(var_dict)
    ys=layer.forward(x)
    if not is_net: ys = [x,ys]
    y = ys[-1]
    if y.dtype != layer.otype:
        print('Warning: output data type not match, %s expected, but get %s! switch to debug mode:'%(layer.otype, y.dtype))
        pdb.set_trace()

    dy=typed_randn(layer.otype, y.shape)
    dv, dx=layer.backward(ys, dy=dy)
    dx_=np.ravel(dx, order='F')
    if eta_x is None:
        eta_x=0.003+0.004j if np.iscomplexobj(x) else 0.005

    res_x=[]
    #check dy/dx
    for i in range(num_check):
        #change variables at random position.
        pos=np.random.randint(0,x.size)
        xn1=x.copy(order='F')
        xn1.ravel(order='F')[pos]+=eta_x/2.
        xn2=x.copy(order='F')
        xn2.ravel(order='F')[pos]-=eta_x/2.
        y1=layer.forward(xn1)
        y2=layer.forward(xn2)
        if is_net: y1=y1[-1]; y2=y2[-1]

        ngrad_x = np.sum((y1-y2)*dy)
        cgrad_x = dx_[pos]*eta_x
        if analytical==2 and layer.itype[:7]=='complex':
            cgrad_x = cgrad_x.real
        diff = abs(cgrad_x-ngrad_x)
        if diff/max(1,abs(cgrad_x))>tol:
            print('Num Diff Test Fail! @x_[%s] = %s'%(pos, x.ravel()[pos]))
            print('XBP Diff = %s, Num Diff = %s'%(cgrad_x, ngrad_x))
            res_x.append(False)
        else:
            res_x.append(True)

    if not is_net and layer.num_variables==0:
        return res_x

    #check dy/dw
    res_w=[]
    var0 = layer.get_variables()
    if eta_w is None:
        eta_w = 0.003+0.004j if np.iscomplexobj(var0) else 0.005
    for i in range(num_check):
        #change variables at random position.
        pos=np.random.randint(0,var0.size)
        var1=var0.copy()
        var1[pos]+=eta_w/2.
        layer.set_variables(var1)
        y1=layer.forward(x)

        var2=var0.copy()
        var2[pos]-=eta_w/2.
        layer.set_variables(var2)
        y2=layer.forward(x)
        if is_net: y1=y1[-1]; y2=y2[-1]

        ngrad_w = np.sum((y1-y2)*dy)
        cgrad_w = dv[pos]*eta_w
        diff=abs(cgrad_w-ngrad_w)
        if diff/max(1, abs(cgrad_w))>tol:
            print('Num Diff Test Fail! @var[%s] = %s'%(pos,var0[pos]))
            print('WBP Diff = %s, Num Diff = %s'%(cgrad_w, ngrad_w))
            res_w.append(False)
        else:
            res_w.append(True)
    return res_w+res_x

def generate_randx(layer):
    '''Generate random input tensor.'''
    max_dim=3
    max_size=20
    input_shape = layer.input_shape
    input_dtype = layer.itype
    if input_shape is None:
        return typed_randn(input_dtype, np.random.randint(1, max_size+1, np.random.randint(1,max_dim+1)))
    input_shape = [n if n>=0 else np.random.randint(1, max_size+1) for n in input_shape]
    return typed_randn(input_dtype, input_shape)

def _check_input(layer, x):
    if layer.input_shape is None or x is None:
        return
    if x.ndim!=len(layer.input_shape):
        raise ValueError('Dimension mismatch! layer %s, x %s, desire %s'%(layer, x.ndim, len(layer.input_shape)))
    for shape_i, xshape_i in zip(layer.input_shape, x.shape):
        if shape_i!=-1 and shape_i!=xshape_i:
            raise ValueError('Illegal Input shape! layer %s, x %s, desire %s'%(layer, x.shape, layer.input_shape))

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
