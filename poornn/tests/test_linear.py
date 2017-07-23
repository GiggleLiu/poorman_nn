'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.linalg import norm,svd
from copy import deepcopy
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import torch.nn as nn
from torch import autograd
import torch
import sys,pdb,time
sys.path.insert(0,'../')

from linears import Linear

def test_linear():
    random.seed(2)
    num_batch=10
    nfi=1024
    nfo=512
    ts=random.randn(num_batch,nfi)
    ts=autograd.Variable(torch.Tensor(ts),requires_grad=True)
    #2 features, kernel size 3x3
    cv=nn.Linear(nfi,nfo)
    fltr=cv.weight.data.numpy()
    sv=Linear(float32(fltr),float32(cv.bias.data.numpy()))
    print "Testing forward for %s"%sv
    xin_np=asfortranarray(ts.data.numpy())
    xin_np1=xin_np[0]
    ntest=5
    t0=time.time()
    for i in xrange(ntest):
        y1=cv(ts)
    t1=time.time()
    for i in xrange(ntest):
        y2=sv.forward(xin_np)
    t2=time.time()
    #for i in xrange(ntest):
        #y3=sv.forward(xin_np1)
    #t3=time.time()
    #print "Elapse old = %s, new = %s, new_1 = %s"%(t1-t0,t2-t1,t3-t2)
    print "Elapse old = %s, new = %s"%(t1-t0,t2-t1)
    res1=y1.data.numpy()
    res2=y2
    #res3=transpose(y3[...,newaxis],(3,2,0,1))
    #assert_allclose(res1,res3,atol=1e-4)
    assert_allclose(res1,res2,atol=1e-4)

    print "Testing backward"
    dy=torch.randn(*y1.size())
    dy_np=asfortranarray(dy.numpy())
    #dy_np1=dy_np[...,0]
    dweight=zeros_like(sv.weight)
    dbias=zeros_like(sv.bias)
    #dweight1=zeros_like(sv._fltr_flatten)
    #dbias1=zeros_like(sv.bias)
    dx=zeros_like(xin_np)
    #dx1=zeros_like(xin_np1)

    t0=time.time()
    y1.backward(dy)
    t1=time.time()
    for i in xrange(ntest):
        (dweight, dbias), dx=sv.backward(xin_np, y2, dy_np, mask=(1,1))
    t2=time.time()
    #for i in xrange(ntest):
        #sv.backward(xin_np1, y3, dy_np1, dx1, dweight1, dbias1, mask=(1,1))
    #t3=time.time()
    #print "Elapse old = %s, new = %s, new-1 = %s"%(t1-t0,(t2-t1)/ntest,(t3-t2)/ntest)
    print "Elapse old = %s, new = %s"%(t1-t0,(t2-t1)/ntest)
    #dweight1/=ntest
    #dx1/=ntest
    #dbias1/=ntest

    #reshape back
    dx0=ts.grad.data.numpy()
    #dx1=dx1[...,newaxis]

    wfactor=dweight.mean()
    bfactor=dbias.mean()
    dweight=reshape(dweight,cv.weight.size())/wfactor
    #dweight1=reshape(dweight1,cv.weight.size())/wfactor
    assert_allclose(dx0,dx,atol=2e-3)
    assert_allclose(cv.weight.grad.data.numpy()/wfactor,dweight,atol=2e-3)
    assert_allclose(cv.bias.grad.data.numpy()/bfactor,dbias/bfactor,atol=3e-3)
    #assert_allclose(dx0,dx1,atol=2e-3)
    #assert_allclose(cv.weight.grad.data.numpy()/wfactor,dweight1,atol=2e-3)
    #assert_allclose(cv.bias.grad.data.numpy()/bfactor,dbias1/bfactor,atol=2e-3)
    pdb.set_trace()

random.seed(2)
test_linear()
