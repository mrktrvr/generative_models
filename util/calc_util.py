#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
calc_util.py
'''
import os

from scipy.misc import logsumexp as scipy_lse
from numpy import array as arr
from numpy import atleast_3d
from numpy import log
from numpy import exp
from numpy import diag
from numpy.linalg import inv as np_inv
from numpy.linalg import slogdet
from numpy.linalg import cholesky
from numpy.linalg import solve

cdir = os.path.abspath(os.path.dirname(__file__))


def inv(src):
    '''
    src: array(d, d) or array(n, d, d)
    dst: array(d, d) or array(n, d, d)
    '''
    dst = np_inv(src)
    return dst


def logdet(src, use_cholesky=True):
    '''
    src = logdet(src)
    src: array(n, d, d)
    dst: float or array(n)
    '''
    if use_cholesky:
        if src.ndim != 3:
            src = atleast_3d(src).transpose(2, 0, 1)
        dst = arr([2 * log(diag(x)).sum() for x in cholesky(src)])
    else:
        ldt, sgn = slogdet(src)
        dst = sgn * ldt
    return dst


def logsumexp(src, axis=None):
    '''
    dst = logsumexp(src, axis=None)
    dst = log(sum(exp(src), axis=axis))
    src: np.array(data_len, [data_dim])
    dst: np.array()
    '''
    dst = scipy_lse(src, axis)
    return dst


def calc_prec_mu(cov, mu):
    '''
    prec: np.array(data_dim, data_dim, n)
    mu: np.array(data_dim, n)x)
    '''
    dst = arr([solve(cov[:, :, k], mu[:, k]) for k in xrange(mu.shape[-1])]).T
    return dst


def main_inv_det():
    from scipy.stats import wishartx
    from numpy import eye
    data_dim = 4
    n_states = 3
    src = wishart.rvs(data_dim, eye(data_dim), size=n_states)
    print 'src:\n', src
    dst = inv(src)
    print 'inv:\n', dst
    dst = logdet(src)
    print 'det:\n', dst


def main_logsumexp():
    from numpy.random import rand
    data_len = 100
    data_dim = 2
    axis = 1
    src = rand(data_len, data_dim)
    dst = logsumexp(src, axis=axis)
    print dst
    print log(exp(src).sum(axis))


if __name__ == '__main__':
    main_inv_det()
    main_logsumexp()
