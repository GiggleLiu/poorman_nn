'''
Tests for MPS and MPO
'''
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import torch
import sys,pdb,time
sys.path.insert(0,'../')

from spconv import SPConv
from core import check_numdiff
from utils import typed_random


