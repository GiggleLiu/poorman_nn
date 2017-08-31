from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time
sys.path.insert(0,'../')

from functions import *
from torch.nn import functional as F
import torch
from checks import check_numdiff
from utils import typed_randn

def test_sigmoid():
    func=Sigmoid((-1,),dtype='complex128')
    print 'Test forward for %s'%func
    xs=array([-1e100,-1e20,-0.5j*pi,-log(2),0,log(2),0.5j*pi,1e20,1e100])
    ys=array([0.,0.,1./(1+1j),1./3,0.5,2./3,1./(1-1j),1.,1.])
    dydx=[0.,0.,1j/(1+1j)**2,2./9,0.25,2./9,-1j/(1-1j)**2,0.,0.]
    assert_allclose(func.forward(xs),ys)
    print 'Test backward for Sigmoid'
    assert_allclose(func.backward([xs,ys],1.)[1],dydx)
    assert_(all(check_numdiff(func, xs)))

def test_log2cosh():
    func=Log2cosh((-1,),dtype='complex128')
    print 'Test forward for %s'%func
    xs=array([-1e100,-1e20,-0.25j*pi,-log(2),0,log(2),0.25j*pi,1e20,1e100])
    ys=array([1e100,1e20,log(2.)/2,log(2.5),log(2.),log(2.5),log(2.)/2,1e20,1e100])
    dydx=tanh(xs)
    assert_allclose(func.forward(xs),ys)
    print 'Test backward'
    assert_allclose(func.backward([xs,ys],1.)[1],dydx)
    #can not pass num check for inifity values
    check_numdiff(func, xs)

def test_pooling():
    for b in ['P','O']:
        for mode in ['max', 'mean']:
            func=Pooling(input_shape=(-1,1,4,4), dtype='complex128', kernel_shape=(2,2), mode=mode, boundary=b)
            print 'Test forward for %s - %sBC'%(func,b)
            x=asfortranarray(arange(16,dtype=func.dtype).reshape([1,1,4,4]))
            dy=asfortranarray(arange(4, dtype='float32').reshape([1,1,2,2]))
            y=func.forward(x)
            if mode=='max':
                y_true_ravel=[5,13,7,15]
                dx_true_ravel=[0,0,0,0,
                    0,0,0,1,
                    0,0,0,0,
                    0,2,0,3]
            else:
                y_true_ravel=[2.5,10.5,4.5,12.5]
                dx_true_ravel=[0,0,0.25,0.25,
                    0,0,0.25,0.25,
                    0.5,0.5,0.75,0.75,
                    0.5,0.5,0.75,0.75]
            assert_allclose(y.ravel(order='F'),y_true_ravel)
            assert_allclose(y.shape,[1,1,2,2])
            print 'Test backward'
            dx=func.backward([x,y],dy)[1]
            assert_allclose(dx.ravel(),dx_true_ravel)
            assert_allclose(dx.shape,[1,1,4,4])
            assert_(all(check_numdiff(func, eta=1e-4)))

def test_pooling_per():
    for mode in ['max', 'max-abs','min','min-abs', 'mean']:
        func=Pooling(input_shape=(10,3,40,40), dtype='complex128', kernel_shape=(2,2), mode=mode)
        print "Num Diff test for %s"%func
        assert_(all(check_numdiff(func, eta=1e-3)))

def test_exp():
    oldshape=(3,4,2)
    dtype='complex128'
    func=Exp(oldshape, dtype)
    xs=typed_randn(dtype, oldshape)
    print 'Test forward for %s'%func
    ys=func.forward(xs)
    assert_allclose(ys,exp(xs))
    print 'Test backward'
    assert_allclose(func.backward([xs,ys],1.)[1],ys)
    assert_(all(check_numdiff(func, xs)))

def test_reshape():
    oldshape=(3,4,2)
    newshape=(3,8)
    dtype='float32'
    func=Reshape(oldshape, newshape, dtype)
    xs=typed_randn(dtype, oldshape)
    print 'Test forward for %s'%func
    ys=func.forward(xs)
    assert_allclose(ys,xs.reshape(newshape, order='F'))
    print 'Test backward'
    dy=typed_randn(dtype, newshape)
    assert_allclose(func.backward([xs,ys],dy)[1],dy.reshape(oldshape,order='F'))
    assert_(all(check_numdiff(func, xs)))

def test_transpose():
    axes=(2,3,1,0)
    func=Transpose((2,3,4,5), 'float32', axes=axes)
    xs=random.random(func.input_shape)
    print 'Test forward for %s'%func
    ys=func.forward(xs)
    assert_allclose(ys,transpose(xs, axes=axes))
    print 'Test backward'
    dy=random.random([4,5,3,2])
    assert_allclose(func.backward([xs,ys],dy)[1],transpose(dy,(3,2,0,1)))
    assert_(all(check_numdiff(func, xs)))

def test_softmax_cross():
    random.seed(2)
    f1=SoftMax(input_shape=(-1,4), dtype='float32',axis=1)
    f2=CrossEntropy(input_shape=(-1,4), dtype='float32', axis=1)
    f3=SoftMaxCrossEntropy(input_shape=(-1,4), dtype='float32', axis=1)
    print 'Test forward for %s, %s, %s.'%(f1,f2,f3)
    x=random.random(12).reshape([3,4], order='F')  #3 batches, 4 logits.
    y_true=array([[0.,1.,0.,0.], [0,0,1.,0],[1,0,0,0]],order='F')
    y1=f1.forward(x)
    rd={'y_true':y_true}
    f2.set_runtime_vars(var_dict=rd)
    f3.set_runtime_vars(var_dict=rd)
    y2=f2.forward(y1)
    y2_=f3.forward(x)
    assert_allclose(y2,y2_)
    print 'Test backward'
    dy2=array([0,1,0.])
    dy1=f2.backward([y1, y2], dy=dy2)[1]
    dx=f1.backward([x,y1],dy1)[1]
    dx_=f3.backward([x,y2_],dy2)[1]
    assert_allclose(dx,dx_)
    assert_(all(check_numdiff(f1, x)))
    assert_(all(check_numdiff(f2, y1, var_dict=rd)))
    assert_(all(check_numdiff(f3, x, var_dict=rd)))

def test_square_loss():
    dtype='float64'
    f3=SquareLoss(input_shape=(-1,4), dtype=dtype)
    print 'Test numdiff for %s.'%(f3,)
    x=typed_randn(dtype, [3,4])  #3 batches, 4 logits.
    y_true=array([[0.,1.,0.,0.], [0,0,1.,0],[1,0,0,0]],order='F')
    rd={'y_true':y_true}
    assert_(all(check_numdiff(f3, x, var_dict=rd)))
 
    dtype='complex128'
    f3=SquareLoss(input_shape=(-1,4), dtype=dtype)
    print 'Test numdiff for complex %s.'%(f3,)
    x=typed_randn(dtype, [3,4])  #3 batches, 4 logits.
    y_true=array([[0.,1.,0.,0.], [0,0,1.,0],[1,0,0,0]],order='F')
    rd={'y_true':y_true}
    print check_numdiff(f3, x, var_dict=rd)

def test_mul():
    dtype='complex128'
    f3=Mul(input_shape=(-1,4), dtype=dtype, alpha=0.3j)
    print 'Test numdiff for %s.'%(f3,)
    #x=typed_randn(dtype, [3,4])  #3 batches, 4 logits.
    assert_(all(check_numdiff(f3)))

def test_power():
    dtype='complex128'
    f3=Power(input_shape=(-1,4), dtype=dtype, order=3.3)
    print 'Test numdiff for %s.'%(f3,)
    assert_(all(check_numdiff(f3)))

def test_convprod():
    dtype='complex128'
    dtype='float32'
    func=ConvProd(input_shape=(3,3), dtype=dtype, kernel_shape=(2,2), boundary='P')
    print 'Test forward for %s.'%(func,)
    x = arange(1,10, dtype='float64').reshape([3,3])
    yt = [[40,180,72],[1120,2160,24*63],[112,432,189]]
    y = func.forward(x)
    assert_allclose(yt,y)
    print 'Test numdiff for %s.'%(func,)
    assert_(all(check_numdiff(func)))

def test_dropout():
    func=DropOut(input_shape=(4,2), dtype='float32', axis=0, keep_rate=0.5)
    func.set_runtime_vars({'seed':2})
    print 'Test forward for %s'%func
    x=arange(8, dtype='float32').reshape([4,2], order='F')
    y=func.forward(x)
    assert_allclose(y,array([[0,4],[1,5],[0,0],[3,7]])*2)
    print 'Test backward'
    dy=arange(8, dtype='float32').reshape([4,2])
    dx=func.backward([x,y],dy)[1]
    assert_allclose(dx,array([[0,1],[2,3],[0,0],[6,7]])*2)

def test_summean():
    random.seed(2)
    func=Sum(input_shape=(-1,2), dtype='float32',axis=1)
    func2=Mean(input_shape=(-1,2), dtype='float32',axis=1)
    print 'Test forward for %s, %s.'%(func, func2)
    x=arange(8).reshape([4,2], order='F')
    y=func.forward(x)
    y2=func2.forward(x)
    assert_allclose(y,[4,6,8,10])
    assert_allclose(y2,y/2.)
    print 'Test backward'
    dy=arange(4,dtype='float32')
    dx=func.backward([x,y],dy)[1]
    dx2=func2.backward([x,y2],dy)[1]
    assert_allclose(dx, reshape([0,1,2,3,0,1,2,3],[4,2], order='F'))
    assert_allclose(dx2, dx/2.)
    assert_(all(check_numdiff(func, x)))
    assert_(all(check_numdiff(func2, x)))

def test_relu_per():
    N1,N2,N3=100,20,10
    x=torch.randn(N1,N2,N3)
    vx=torch.autograd.Variable(x)
    vx.requires_grad=True
    y0=F.relu(vx)

    func=ReLU(x.size(), 'float32', 0.)
    print 'Test forward for %s'%func
    y=func.forward(asfortranarray(x.numpy()))
    assert_allclose(y0.data.numpy(),y)
    print 'Test backward'
    dy=torch.randn(N1,N2,N3)
    y0.backward(gradient=dy)
    dx=func.backward([x.numpy(),y],dy.numpy())[1]
    assert_allclose(dx, vx.grad.data.numpy())

def test_softmax_cross_per():
    N1,N3=100,10
    x=torch.randn(N1,N3)
    y_true=torch.from_numpy(random.randint(0,N3,N1))
    vy_true=torch.autograd.Variable(y_true)
    vx=torch.autograd.Variable(x)
    vx.requires_grad=True
    #y0s=F.softmax(vx)  #to the last dimension
    y0s=F.log_softmax(vx)  #to the last dimension
    y0c=F.nll_loss(y0s,vy_true)  #to the last dimension
    y0c_one=F.cross_entropy(vx, vy_true)

    f1=SoftMax(input_shape=(N1,N3),dtype='float32',axis=1)
    f2=CrossEntropy(input_shape=(N1,N3),dtype='float32',axis=1)
    f3=SoftMaxCrossEntropy(input_shape=(N1,N3), dtype='float32',axis=1)
    f4=Mean(input_shape=(N1,), dtype='float32',axis=0)
    print 'Test forward for %s, %s, %s, %s'%(f1,f2,f3,f4)
    x_np=x.numpy()
    y_true_hot=zeros(f1.output_shape)
    y_true_hot[arange(N1),y_true.numpy()]=1
    y1=f1.forward(x_np)
    rd={'y_true':y_true_hot}
    f2.set_runtime_vars(var_dict=rd)
    f3.set_runtime_vars(var_dict=rd)
    y2=f2.forward(y1)
    y2_=f3.forward(x_np)
    assert_allclose(y2,y2_,atol=1e-5)
    y4=f4.forward(y2)
    assert_allclose(y4,y0c.data.numpy(),atol=1e-5)

    print 'Test backward'
    dyc=torch.rand(1)
    y0c.backward(dyc)

    dy4=dyc.numpy()[0]
    dy2=f4.backward([y2,y4],dy4)[1]
    dy1=f2.backward([y1, y2], dy=dy2)[1]
    dx=f1.backward([x_np,y1],dy1)[1]
    dx_=f3.backward([x_np,y2_],dy2)[1]
    assert_allclose(dx,dx_,atol=1e-5)
    assert_allclose(dx,vx.grad.data.numpy(),atol=1e-5)
    assert_(all(check_numdiff(f1, x_np)))
    assert_(all(check_numdiff(f2, y1, var_dict=rd)))
    assert_(all(check_numdiff(f3, x_np, var_dict=rd)))
    assert_(all(check_numdiff(f4, y2)))

def test_all():
    random.seed(3)
    torch.manual_seed(3)
    test_convprod()
    test_pooling_per()
    test_square_loss()
    test_mul()
    test_power()
    test_reshape()
    test_exp()
    test_transpose()
    test_softmax_cross_per()
    test_relu_per()
    test_summean()
    test_softmax_cross()
    test_dropout()
    test_pooling()
    test_log2cosh()
    test_sigmoid()

test_all()
