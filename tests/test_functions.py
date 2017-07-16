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
        x=arange(16).reshape([4,4,1,1])
        y=func.forward(x)
        assert_allclose(y.ravel(),[5,7,13,15])
        assert_allclose(y.shape,[2,2,1,1])
        print 'Test backward'
        dy=arange(4).reshape([2,2,1,1])
        dx=func.backward(x,y,dy)[1]
        assert_allclose(dx.ravel(),[0,0,0,0,
            0,0,0,1,
            0,0,0,0,
            0,2,0,3])
        assert_allclose(dx.shape,[4,4,1,1])

def test_all():
    test_maxpool()
    test_sigmoid()
    test_log2cosh()

test_all()
