from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time
sys.path.insert(0,'../')

from functions import *
from torch.nn import functional as F
import torch

def test_sigmoid():
    func=Sigmoid()
    print 'Test forward for %s'%func
    xs=array([-1e100,-1e20,-0.5j*pi,-log(2),0,log(2),0.5j*pi,1e20,1e100])
    ys=array([0.,0.,1./(1+1j),1./3,0.5,2./3,1./(1-1j),1.,1.])
    dydx=[0.,0.,1j/(1+1j)**2,2./9,0.25,2./9,-1j/(1-1j)**2,0.,0.]
    assert_allclose(func.forward(xs),ys)
    print 'Test backward for Sigmoid'
    assert_allclose(func.backward(xs,ys,1.)[1],dydx)

def test_log2cosh():
    func=Log2cosh()
    print 'Test forward for %s'%func
    xs=array([-1e100,-1e20,-0.25j*pi,-log(2),0,log(2),0.25j*pi,1e20,1e100])
    ys=array([1e100,1e20,log(2.)/2,log(2.5),log(2.),log(2.5),log(2.)/2,1e20,1e100])
    dydx=tanh(xs)
    assert_allclose(func.forward(xs),ys)
    print 'Test backward'
    assert_allclose(func.backward(xs,ys,1.)[1],dydx)

def test_maxpool():
    for b in ['P','O']:
        func=MaxPool(input_shape=(-1,1,4,4), output_shape=(-1,1,2,2), kernel_shape=(2,2), boundary='O')
        print 'Test forward for %s - %sBC'%(func,b)
        x=arange(16).reshape([1,1,4,4], order='F')
        y=func.forward(x)
        assert_allclose(y.ravel(),[5,13,7,15])
        assert_allclose(y.shape,[1,1,2,2])
        print 'Test backward'
        dy=arange(4).reshape([1,1,2,2])
        dx=func.backward(x,y,dy)[1]
        assert_allclose(dx.ravel(),[0,0,0,0,
            0,0,0,1,
            0,0,0,0,
            0,2,0,3])
        assert_allclose(dx.shape,[1,1,4,4])

def test_exp():
    oldshape=(3,4,2)
    func=Exp()
    xs=random.random(oldshape)
    print 'Test forward for %s'%func
    ys=func.forward(xs)
    assert_allclose(ys,exp(xs))
    print 'Test backward'
    assert_allclose(func.backward(xs,ys,1.)[1],ys)

def test_reshape():
    oldshape=(3,4,2)
    newshape=(3,8)
    func=Reshape(oldshape, newshape)
    xs=random.random(oldshape)
    print 'Test forward for %s'%func
    ys=func.forward(xs)
    assert_allclose(ys,xs.reshape(newshape))
    print 'Test backward'
    dy=random.random(newshape)
    assert_allclose(func.backward(xs,ys,dy)[1],dy.reshape(oldshape))

def test_transpose():
    axes=(2,3,1,0)
    func=Transpose((2,3,4,5), axes=axes)
    xs=random.random(func.input_shape)
    print 'Test forward for %s'%func
    ys=func.forward(xs)
    assert_allclose(ys,transpose(xs, axes=axes))
    print 'Test backward'
    dy=random.random([4,5,3,2])
    assert_allclose(func.backward(xs,ys,dy)[1],transpose(dy,(3,2,0,1)))


def test_softmax_cross():
    random.seed(2)
    f1=SoftMax(input_shape=(-1,4),axis=1)
    f2=CrossEntropy(input_shape=(-1,4),axis=1)
    f3=SoftMaxCrossEntropy(input_shape=(-1,4), axis=1)
    print 'Test forward for %s, %s, %s.'%(f1,f2,f3)
    x=random.random(12).reshape([3,4], order='F')  #3 batches, 4 logits.
    y_true=random.randint(0,2,[4,3]).T
    y1=f1.forward(x)
    f2.set_y_true(y_true)
    f3.set_y_true(y_true)
    y2=f2.forward(y1)
    y2_=f3.forward(x)
    assert_allclose(y2,y2_)
    print 'Test backward'
    dy2=array([0,1,0.])
    dy1=f2.backward(y1, y2, dy=dy2)[1]
    dx=f1.backward(x,y1,dy1)[1]
    dx_=f3.backward(x,y2_,dy2)[1]
    assert_allclose(dx,dx_)

def test_dropout():
    random.seed(2)
    func=DropOut_I(input_shape=(-1,2), axis=0, keep_rate=0.5)
    print 'Test forward for %s'%func
    x=arange(8, dtype='float32').reshape([4,2], order='F')
    y=func.forward(x)
    assert_allclose(y,array([[0,4],[1,5],[0,0],[3,7]])*2)
    print 'Test backward'
    dy=arange(8, dtype='float32').reshape([4,2])
    dx=func.backward(x,y,dy)[1]
    assert_allclose(dx,array([[0,1],[2,3],[0,0],[6,7]])*2)

def test_summean():
    random.seed(2)
    func=Sum(input_shape=(-1,2),axis=1)
    func2=Mean(input_shape=(-1,2),axis=1)
    print 'Test forward for %s, %s.'%(func, func2)
    x=arange(8).reshape([4,2], order='F')
    y=func.forward(x)
    y2=func2.forward(x)
    assert_allclose(y,[4,6,8,10])
    assert_allclose(y2,y/2.)
    print 'Test backward'
    dy=arange(4,dtype='float32')
    dx=func.backward(x,y,dy)[1]
    dx2=func2.backward(x,y2,dy)[1]
    assert_allclose(dx, reshape([0,1,2,3,0,1,2,3],[4,2], order='F'))
    assert_allclose(dx2, dx/2.)

def test_relu():
    func=ReLU_I(0.1)
    print 'Test forward for %s'%func
    x=arange(-3,5, dtype='float32').reshape([4,2], order='F')
    y=func.forward(x)
    assert_allclose(y,[[-0.3,1],[-0.2,2],[-0.1,3],[0,4]])
    print 'Test backward'
    dy=arange(-2,6,dtype='float32').reshape([4,2],order='F')
    dx=func.backward(x,y,dy)[1]
    assert_allclose(dx, [[-0.2,2],[-0.1,3],[0,4],[0.1,5]])

def test_relu_per():
    N1,N2,N3=100,20,10
    x=torch.randn(N1,N2,N3)
    vx=torch.autograd.Variable(x)
    vx.requires_grad=True
    y0=F.relu(vx)

    func=ReLU_I(0.)
    print 'Test forward for %s'%func
    y=func.forward(x.numpy())
    assert_allclose(y0.data.numpy(),y)
    print 'Test backward'
    dy=torch.randn(N1,N2,N3)
    y0.backward(gradient=dy)
    dx=func.backward(x.numpy(),y,dy.numpy())[1]
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

    f1=SoftMax(input_shape=(N1,N3),axis=1)
    f2=CrossEntropy(input_shape=(N1,N3),axis=1)
    f3=SoftMaxCrossEntropy(input_shape=(N1,N3), axis=1)
    f4=Mean(input_shape=(N1,), axis=0)
    print 'Test forward for %s, %s, %s, %s'%(f1,f2,f3,f4)
    x_np=x.numpy()
    y_true_hot=zeros(f1.output_shape)
    y_true_hot[arange(N1),y_true.numpy()]=1
    y1=f1.forward(x_np)
    f2.set_y_true(y_true_hot)
    f3.set_y_true(y_true_hot)
    y2=f2.forward(y1)
    y2_=f3.forward(x_np)
    assert_allclose(y2,y2_,atol=1e-5)
    y4=f4.forward(y2)
    assert_allclose(y4,y0c.data.numpy(),atol=1e-5)

    print 'Test backward'
    dyc=torch.rand(1)
    y0c.backward(dyc)

    dy4=dyc.numpy()[0]
    dy2=f4.backward(y2,y4,dy4)[1]
    dy1=f2.backward(y1, y2, dy=dy2)[1]
    dx=f1.backward(x_np,y1,dy1)[1]
    dx_=f3.backward(x_np,y2_,dy2)[1]
    assert_allclose(dx,dx_,atol=1e-5)
    assert_allclose(dx,vx.grad.data.numpy(),atol=1e-5)

def test_all():
    test_reshape()
    test_exp()
    test_transpose()
    test_softmax_cross_per()
    test_relu_per()
    test_relu()
    test_summean()
    test_softmax_cross()
    test_dropout()
    test_maxpool()
    test_log2cosh()
    test_sigmoid()

test_all()
