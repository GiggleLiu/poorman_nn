from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time
sys.path.insert(0,'../')

from functions import *

def test_sigmoid():
    print 'Test forward for Sigmoid'
    func=Sigmoid()
    xs=array([-1e100,-1e20,-0.5j*pi,-log(2),0,log(2),0.5j*pi,1e20,1e100])
    ys=array([0.,0.,1./(1+1j),1./3,0.5,2./3,1./(1-1j),1.,1.])
    dydx=[0.,0.,1j/(1+1j)**2,2./9,0.25,2./9,-1j/(1-1j)**2,0.,0.]
    assert_allclose(func.forward(xs),ys)
    print 'Test backward for Sigmoid'
    assert_allclose(func.backward(xs,ys,1.)[1],dydx)

def test_log2cosh():
    print 'Test forward for Log2cosh'
    func=Log2cosh()
    xs=array([-1e100,-1e20,-0.25j*pi,-log(2),0,log(2),0.25j*pi,1e20,1e100])
    ys=array([1e100,1e20,log(2.)/2,log(2.5),log(2.),log(2.5),log(2.)/2,1e20,1e100])
    dydx=tanh(xs)
    assert_allclose(func.forward(xs),ys)
    print 'Test backward'
    assert_allclose(func.backward(xs,ys,1.)[1],dydx)

def test_maxpool():
    for b in ['P','O']:
        print 'Test maxpool forward - %sBC'%b
        func=MaxPool(input_shape=(-1,1,4,4), output_shape=(-1,1,2,2), kernel_shape=(2,2), boundary='O')
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

def test_losts():
    random.seed(2)
    print 'Test lost functions'
    f1=SoftMax(input_shape=(-1,4),axis=-1)
    f2=CrossEntropy(input_shape=(-1,4),axis=-1)
    f3=SoftMaxCrossEntropy(input_shape=(-1,4), axis=-1)
    x=random.random(12).reshape([3,4], order='F')  #3 batches, 4 logits.
    y_true=random.randint(0,2,[4,3]).T
    y1=f1.forward(x)
    f2.set_y_true(y_true)
    f3.set_y_true(y_true)
    y2=f2.forward(y1)
    y2_=f3.forward(x)
    assert_allclose(y2,y2_)
    print 'Test backward'
    dy2=ones(3, dtype=float32)
    dy1=f2.backward(y1, y2, dy=dy2)[1]
    dx=f1.backward(x,y1,dy1)[1]
    dx_=f3.backward(x,y2_,dy2)[1]
    assert_allclose(dx,dx_)

def test_dropout():
    random.seed(2)
    print 'Test DropOut forward'
    func=DropOut(input_shape=(-1,2), axis=0, keep_rate=0.5)
    x=arange(8, dtype='float32').reshape([4,2], order='F')
    y=func.forward(x)
    assert_allclose(y,array([[0,4],[1,5],[0,0],[3,7]])*2)
    print 'Test backward'
    dy=arange(8, dtype='float32').reshape([4,2])
    dx=func.backward(x,y,dy)[1]
    assert_allclose(dx,array([[0,1],[2,3],[0,0],[6,7]])*2)

def test_summean():
    random.seed(2)
    print 'Test forward for Sum/Mean.'
    func=Sum(input_shape=(-1,2),axis=1)
    func2=Mean(input_shape=(-1,2),axis=1)
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
    func=ReLU(0.1)
    print 'Test forward for ReLU.'
    x=arange(-3,5, dtype='float32').reshape([4,2], order='F')
    y=func.forward(x)
    assert_allclose(y,[[-0.3,1],[-0.2,2],[-0.1,3],[0,4]])
    print 'Test backward'
    dy=arange(-2,6,dtype='float32').reshape([4,2],order='F')
    dx=func.backward(x,y,dy)[1]
    assert_allclose(dx, [[-0.2,2],[-0.1,3],[0,4],[1,5]])

def test_all():
    test_relu()
    test_summean()
    test_losts()
    test_dropout()
    test_maxpool()
    test_log2cosh()
    test_sigmoid()

test_all()
