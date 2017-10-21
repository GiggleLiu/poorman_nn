'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.testing import dec, assert_, assert_raises,\
    assert_almost_equal, assert_allclose
from scipy import sparse as sps
import pdb
import time
import pytest

from ..spconv import SPConv, SPSP
from ..checks import check_numdiff
from ..utils import typed_randn

random.seed(2)
try:
    import torch
    torch.manual_seed(2)
except:
    print('Skip Comparative Benchmark with Pytorch!')


def test_conv2d():
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn import Conv1d, Conv2d
        from torch import autograd
    except:
        return
    ts = arange(16, dtype='float32').reshape([1, 4, 4])
    # [ 0, 1, 2, 3]
    # [ 4, 5, 6, 7]
    # [ 8, 9,10,11]
    # [12,13,14,15]
    ts = autograd.Variable(torch.Tensor(array([ts, ts + 1.0])))
    # 2 features, kernel size 3x3
    cv = Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
    cv.weight.data[...] = torch.Tensor([[[[1., 1, 1], [1, 0, 1], [1, 1, 1.]]],
                                        [[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]])
    cv.bias.data[...] = torch.Tensor([0, 1])
    res = cv(ts)
    assert_allclose(res.data.numpy(), [[[[40, 48], [72, 80]],
                                        [[21, 25], [37, 41]]], [
        [[48, 56], [80, 88]], [[25, 29], [41, 45]]]])

    # new
    sv = SPConv((2, 1, 4, 4), 'complex128', weight=cv.weight.data.numpy(
    ), bias=cv.bias.data.numpy(), strides=(1, 1), boundary='O')
    x = asfortranarray(ts.data.numpy(), dtype='complex128')
    res2 = sv.forward(x)
    assert_allclose(res2, [[[[40, 48], [72, 80]], [[21, 25], [37, 41]]], [
                    [[48, 56], [80, 88]], [[25, 29], [41, 45]]]])


def test_conv2d_per():
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn import Conv1d, Conv2d
        from torch import autograd
    except:
        return
    num_batch = 1
    dim_x = 30
    dim_y = 40
    K1 = 3
    K2 = 4
    nfin = 10
    nfout = 16
    ts = random.random([num_batch, nfin, dim_x, dim_y])
    ts = autograd.Variable(torch.Tensor(ts), requires_grad=True)
    # 2 features, kernel size 3x3
    cv = Conv2d(nfin, nfout, kernel_size=(K1, K2),
                stride=(1, 1), padding=(0, 0))
    weight = cv.weight.data.numpy()
    sv = SPConv((-1, nfin, dim_x, dim_y), 'float32', float32(weight),
                float32(cv.bias.data.numpy()), strides=(1, 1),
                boundary='O', w_contiguous=True)
    sv2 = SPConv((nfin, dim_x, dim_y), 'float32', float32(weight), float32(
        cv.bias.data.numpy()), strides=(1, 1), boundary='O', w_contiguous=True)
    print("Testing forward for %s, 2D" % sv)
    xin_np = ts.data.numpy()
    xin_np1 = xin_np[0]
    ntest = 5
    t0 = time.time()
    for i in range(ntest):
        y1 = cv(ts)
    t1 = time.time()
    for i in range(ntest):
        y2 = sv.forward(xin_np)
    t2 = time.time()
    for i in range(ntest):
        y3 = sv2.forward(xin_np1)
    t3 = time.time()
    print("Elapse old = %s, new = %s, new_1 = %s" %
          (t1 - t0, t2 - t1, t3 - t2))
    res1 = y1.data.numpy()
    res2 = y2
    res3 = y3[newaxis]
    assert_allclose(res1, res2, atol=1e-4)
    assert_allclose(res1, res3, atol=1e-4)

    print("Testing backward")
    dy = torch.randn(*y1.size())
    dy_np = asfortranarray(dy.numpy())
    dy_np1 = dy_np[0]

    t0 = time.time()
    y1.backward(dy)
    t1 = time.time()
    for i in range(ntest):
        dwb, dx = sv.backward([xin_np, y2], dy_np)
    t2 = time.time()
    for i in range(ntest):
        dwb1, dx1 = sv2.backward([xin_np1, y3], dy_np1)
    t3 = time.time()
    print("Elapse old = %s, new = %s, new-1 = %s" %
          (t1 - t0, (t2 - t1) / ntest, (t3 - t2) / ntest))

    # reshape back
    dx0 = ts.grad.data.numpy()
    dx1 = dx1[newaxis]

    dweight, dbias = dwb[:sv.weight.size], dwb[sv.weight.size:]
    dweight1, dbias1 = dwb1[:sv2.weight.size], dwb1[sv.weight.size:]
    wfactor = dweight.mean()
    bfactor = dbias.mean()
    dweight = reshape(dweight, cv.weight.size(), order='F') / wfactor
    dweight1 = reshape(dweight1, cv.weight.size(), order='F') / wfactor
    assert_allclose(cv.bias.grad.data.numpy() /
                    bfactor, dbias / bfactor, atol=2e-3)
    assert_allclose(cv.bias.grad.data.numpy() / bfactor,
                    dbias1 / bfactor, atol=2e-3)

    assert_allclose(dx0, dx, atol=2e-3)
    assert_allclose(dx0, dx1, atol=2e-3)

    assert_allclose(cv.weight.grad.data.numpy() / wfactor, dweight1, atol=2e-3)
    assert_allclose(cv.weight.grad.data.numpy() / wfactor, dweight, atol=2e-3)
    assert_(all(check_numdiff(sv, xin_np)))


def test_conv1d_per():
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn import Conv1d, Conv2d
        from torch import autograd
    except:
        return
    num_batch = 1
    dim_x = 30
    K1 = 3
    nfin = 10
    nfout = 16
    ts = random.random([num_batch, nfin, dim_x])
    ts = autograd.Variable(torch.Tensor(ts), requires_grad=True)
    # 2 features, kernel size 3x3
    cv = Conv2d(nfin, nfout, kernel_size=(K1,), stride=(1,), padding=(0,))
    weight = cv.weight.data.numpy()
    sv = SPConv((-1, nfin, dim_x), 'float32', float32(weight),
                float32(cv.bias.data.numpy()), strides=(1,),
                boundary='O', w_contiguous=True)
    sv2 = SPConv((nfin, dim_x), 'float32', float32(weight), float32(
        cv.bias.data.numpy()), strides=(1,), boundary='O', w_contiguous=True)
    print("Testing forward for %s, 1D" % sv)
    xin_np = asfortranarray(ts.data.numpy())
    xin_np1 = xin_np[0]
    ntest = 5
    t0 = time.time()
    for i in range(ntest):
        y1 = cv(ts)
    t1 = time.time()
    for i in range(ntest):
        y2 = sv.forward(xin_np)
    t2 = time.time()
    for i in range(ntest):
        y3 = sv2.forward(xin_np1)
    t3 = time.time()
    print("Elapse old = %s, new = %s, new_1 = %s" %
          (t1 - t0, t2 - t1, t3 - t2))
    res1 = y1.data.numpy()
    res2 = y2
    res3 = y3[newaxis]
    assert_allclose(res1, res2, atol=1e-4)
    assert_allclose(res1, res3, atol=1e-4)

    print("Testing backward")
    dy = torch.randn(*y1.size())
    dy_np = asfortranarray(dy.numpy())
    dy_np1 = dy_np[0]

    t0 = time.time()
    y1.backward(dy)
    t1 = time.time()
    for i in range(ntest):
        dwb, dx = sv.backward([xin_np, y2], dy_np)
    t2 = time.time()
    for i in range(ntest):
        dwb1, dx1 = sv2.backward([xin_np1, y3], dy_np1)
    t3 = time.time()
    print("Elapse old = %s, new = %s, new-1 = %s" %
          (t1 - t0, (t2 - t1) / ntest, (t3 - t2) / ntest))

    # reshape back
    dx0 = ts.grad.data.numpy()
    dx1 = dx1[newaxis]

    dweight, dbias = dwb[:sv.weight.size], dwb[sv.weight.size:]
    dweight1, dbias1 = dwb1[:sv2.weight.size], dwb1[sv.weight.size:]
    wfactor = dweight.mean()
    bfactor = dbias.mean()
    dweight = reshape(dweight, cv.weight.size(), order='F') / wfactor
    dweight1 = reshape(dweight1, cv.weight.size(), order='F') / wfactor
    assert_allclose(cv.bias.grad.data.numpy() /
                    bfactor, dbias / bfactor, atol=2e-3)
    assert_allclose(cv.bias.grad.data.numpy() / bfactor,
                    dbias1 / bfactor, atol=2e-3)

    assert_allclose(dx0, dx, atol=2e-3)
    assert_allclose(dx0, dx1, atol=2e-3)

    assert_allclose(cv.weight.grad.data.numpy() / wfactor, dweight1, atol=2e-3)
    assert_allclose(cv.weight.grad.data.numpy() / wfactor, dweight, atol=2e-3)
    assert_(all(check_numdiff(sv, xin_np)))


def test_conv2d_complex():
    num_batch = 1
    dim_x = 10
    dim_y = 20
    K1 = 3
    K2 = 4
    nfin = 4
    nfout = 6
    xin_np = asfortranarray(typed_randn(
        'complex128', [num_batch, nfin, dim_x, dim_y]))
    weight = asfortranarray(typed_randn('complex128', [nfout, nfin, K1, K2]))
    bias = typed_randn('complex128', [nfout])
    sv = SPConv((-1, nfin, dim_x, dim_y), 'complex128', weight,
                bias, strides=(1, 1), boundary='O', w_contiguous=True)
    sv2 = SPConv((nfin, dim_x, dim_y), 'complex128', weight, bias,
                 strides=(1, 1), boundary='O', w_contiguous=True)
    print("Testing forward for %s" % sv)
    xin_np1 = xin_np[0]
    ntest = 5
    t1 = time.time()
    for i in range(ntest):
        y2 = sv.forward(xin_np)
    t2 = time.time()
    for i in range(ntest):
        y3 = sv2.forward(xin_np1)
    t3 = time.time()
    print("Elapse new = %s, new_1 = %s" % (t2 - t1, t3 - t2))
    res2 = y2
    res3 = y3[newaxis]
    assert_allclose(res2, res2, atol=1e-4)

    print("Testing backward")
    dy_np = typed_randn('complex128', y2.shape)
    dy_np1 = dy_np[0]

    t1 = time.time()
    for i in range(ntest):
        dwb, dx = sv.backward([xin_np, y2], dy_np)
    t2 = time.time()
    for i in range(ntest):
        dwb1, dx1 = sv2.backward([xin_np1, y3], dy_np1)
    t3 = time.time()
    print("Elapse new = %s, new-1 = %s" %
          ((t2 - t1) / ntest, (t3 - t2) / ntest))

    # reshape back
    dx1 = dx1[newaxis]

    dweight, dbias = dwb[:sv.weight.size], dwb[sv.weight.size:]
    dweight1, dbias1 = dwb1[:sv2.weight.size], dwb1[sv.weight.size:]
    dweight1 = reshape(dweight1, dweight.shape, order='F')

    assert_allclose(dbias, dbias1, atol=1e-3)
    assert_allclose(dx, dx1, atol=1e-3)
    assert_allclose(dweight1, dweight, atol=1e-3)

    assert_(all(check_numdiff(sv, num_check=100)))
    assert_(all(check_numdiff(sv2, num_check=100)))


@pytest.mark.skip
def test_spsp_complex():
    num_batch = 1
    dim_x = 10
    dim_y = 20
    K1 = 3
    K2 = 4
    nfin = 4
    nfout = 6
    dtype = 'complex128'
    nsite = dim_x * dim_y
    xin_np = asfortranarray(typed_randn(
        dtype, [num_batch, nfin, dim_x, dim_y]))
    weight = asfortranarray(typed_randn(dtype, [nfout, nfin, K1, K2]))
    bias = typed_randn(dtype, [nfout])
    sv = SPConv((-1, nfin, dim_x, dim_y), dtype, weight, bias,
                strides=(1, 1), boundary='P', w_contiguous=True)

    # the corresponding SPSP matrix
    dmat = zeros((nfin, dim_x, dim_y, nsite * nfout), dtype=dtype)

    fltr_ = transpose(weight, (1, 2, 3, 0))
    for i in range(dim_x):
        for j in range(dim_y):
            k = i * dim_y + j
            ox = max(0, i + K1 - dim_x + 1)
            oy = max(0, j + K2 - dim_y + 1)
            dmat[:, i:i + K1 - ox, j:j + K2 - oy, k *
                 nfout:(k + 1) * nfout] = fltr_[:, :K1 - ox, :K2 - oy]
            dmat[:, :ox, :oy, k * nfout:(k + 1) *
                 nfout] = fltr_[:, K1 - ox:, K2 - oy:]
    cscmat = sps.csc_matrix(dmat.reshape(nfin * nsite, nfout * nsite))
    sv2 = SPSP((-1, nfin, dim_x, dim_y), dtype, cscmat, bias, strides=(1, 1))
    print("Testing forward for %s" % sv)
    xin_np1 = xin_np[0]
    ntest = 5
    t1 = time.time()
    for i in range(ntest):
        y2 = sv.forward(xin_np)
    t2 = time.time()
    for i in range(ntest):
        y3 = sv2.forward(xin_np1)
    t3 = time.time()
    print("Elapse new = %s, new-sp = %s" % (t2 - t1, t3 - t2))
    res2 = y2
    res3 = y3[newaxis]
    assert_allclose(res2, res2, atol=1e-4)

    print("Testing backward")
    dy_np = typed_randn(dtype, y2.shape)
    dy_np1 = dy_np[0]

    t1 = time.time()
    for i in range(ntest):
        dwb, dx = sv.backward([xin_np, y2], dy_np)
    t2 = time.time()
    for i in range(ntest):
        dwb1, dx1 = sv2.backward([xin_np1, y3], dy_np1)
    t3 = time.time()
    print("Elapse new = %s, new-sp = %s" %
          ((t2 - t1) / ntest, (t3 - t2) / ntest))

    # reshape back
    dx1 = dx1[newaxis]

    dweight, dbias = dwb[:sv.weight.size], dwb[sv.weight.size:]
    dweight1, dbias1 = dwb1[:sv2.weight.size], dwb1[sv.weight.size:]
    dweight1 = reshape(dweight1, dweight.shape, order='F')

    assert_allclose(dbias, dbias1, atol=1e-3)
    assert_allclose(dx, dx1, atol=1e-3)
    assert_allclose(dweight1, dweight, atol=1e-3)

    assert_(all(check_numdiff(sv, num_check=100)))
    assert_(all(check_numdiff(sv2, num_check=100)))


def run_all():
    # test_spsp_complex()
    test_conv2d_complex()
    test_conv2d()
    test_conv2d_per()
    test_conv1d_per()


if __name__ == '__main__':
    run_all()
