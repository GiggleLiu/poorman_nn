from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time
sys.path.insert(0,'../')

from functions import *

def test_sigmoid():
    func=Sigmoid()
    xs=[-1e100,-1e20,-0.5j*pi,-log(2),0,log(2),0.5j*pi,1e20,1e100]
    ys=[0.,0.,1./(1+1j),1./3,0.5,2./3,1./(1-1j),1.,1.]
    dydx=[0.,0.,1j/(1+1j)**2,2./9,0.25,2./9,-1j/(1-1j)**2,0.,0.]
    assert_allclose(func.forward(xs),ys)
    assert_allclose(func.backward(xs,ys)[1],dydx)
    assert_allclose([func(x) for x in xs],ys)

def test_log2cosh():
    func=Log2cosh()
    xs=[-1e100,-1e20,-0.25j*pi,-log(2),0,log(2),0.25j*pi,1e20,1e100]
    ys=[1e100,1e20,log(2.)/2,log(2.5),log(2.),log(2.5),log(2.)/2,1e20,1e100]
    dydx=tanh(xs)
    assert_allclose(func.forward(xs),ys)
    assert_allclose(func.backward(xs,ys)[1],dydx)
    assert_allclose([func(x) for x in xs],ys)

def test_maxpool():
    for b in ['P','O']:
        print 'Test maxpool forward - %sBC'%b
        func=MaxPool(kernel_shape=(2,2), img_in_shape=(4,4), boundary='O')
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
    f1=SoftMax()
    f2=CrossEntropy()
    f3=SoftMaxCrossEntropy()
    x=random.random(12).reshape([3,4], order='F')  #3 batches, 4 logits.
    y_true=random.randint(0,2,[4,3]).T
    y1=f1.forward(x)
    y2=f2.forward(y1, y_true)
    y2_=f3.forward(x, y_true)
    assert_allclose(y2,y2_)
    print 'Test backward'
    dy2=ones(3, dtype=float32)
    dy1=f2.backward(y1, y2, dy=dy2, y_true=y_true)[1]
    dx=f1.backward(x,y1,dy1)[1]
    dx_=f3.backward(x,y2_,dy2,y_true)[1]
    assert_allclose(dx,dx_)

def test_dropout():
    random.seed(2)
    print 'Test dropout forward'
    func=DropOut(keep_rate=0.5)
    x=arange(8).reshape([4,2], order='F')
    y=func.forward(x)
    assert_allclose(y.ravel(),[0,4,1,5,3,7])
    assert_allclose(y.shape,[3,2])
    print 'Test backward'
    dy=arange(6).reshape([3,2])
    dx=func.backward(x,y,dy)[1]
    assert_allclose(dx,reshape([0,1,2,3,0,0,4,5],[4,2]))

def test_all():
    test_losts()
    test_dropout()
    test_maxpool()
    test_sigmoid()
    test_log2cosh()

test_all()
