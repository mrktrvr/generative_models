#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
calc_util.py
'''
from numpy import array as arr
from numpy import log
from numpy import exp
from numpy import einsum

from .logger import logger


def inv(src):
    '''
    src: array(d, d) or array(n, d, d)
    dst: array(d, d) or array(n, d, d)
    '''
    from numpy.linalg import inv as np_inv
    dst = np_inv(src)
    return dst


def calc_lndet(src):
    '''
    src = logdet(src)
    src: array(d, d) or array(n, d, d) or array(l, n, d, d)
    dst: float or array(n)
    '''
    if src.ndim == 1:
        dst = src
    elif src.ndim == 2:
        dst = logdet(src)
    elif src.ndim == 3:
        dst = logdet(src.transpose(2, 0, 1))
    elif src.ndim == 4:
        dst = logdet(src.transpose(2, 3, 0, 1))
    else:
        logger.error('src.ndim %d not supported' % src.ndima)
        dst = None
    return dst


def logdet(src):
    '''
    src = logdet(src)
    src: array(d, d) or array(n, d, d) or array(l, n, d, d)
    dst: float or array(n)
    '''
    try:
        from numpy.linalg import cholesky
        if src.ndim == 2:
            dst = 2 * log(einsum('dd->d', cholesky(src))).sum()
        if src.ndim == 3:
            dst = 2 * log(einsum('ldd->ld', cholesky(src))).sum(1)
        elif src.ndim == 4:
            dst = 2 * log(einsum('lkdd->lkd', cholesky(src))).sum(2)
        else:
            logger.error('dim %d is not supprted.' % src.ndim)
    except Exception as e:
        from numpy.linalg import slogdet
        logger.info('%s. shape=%s' % (e, list(src.shape)))
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
    from scipy.special import logsumexp as scipy_lse
    dst = scipy_lse(src, axis)
    return dst


def calc_prec_mu(cov, mu):
    '''
    prec: np.array(data_dim, data_dim, n)
    mu: np.array(data_dim, n)x)
    '''
    from numpy.linalg import solve
    dst = arr([solve(cov[:, :, k], mu[:, k]) for k in range(mu.shape[-1])]).T
    return dst


def logmatprod(ln_a, ln_b):
    '''
    ln_[i, j] = log(sum(exp(ln a[i, ...] + ln b[:, j])))
    parameters
    ln_a: np.array(size_A, ...)
    ln_b: np.array(size_A, size_B)

    returns
    ln_C: np.array(size_A, size_B)
    '''
    from numpy import zeros
    ln_a = arr([ln_a]) if ln_a.ndim == 1 else ln_a
    ln_b = arr([ln_b]) if ln_b.ndim == 1 else ln_b
    I = ln_a.shape[0]
    J = ln_b.shape[1]
    ln_C = zeros((I, J))
    for i in range(I):
        for j in range(J):
            ln_C[i, j] = logsumexp(ln_a[i] + ln_b[:, j], -1)
    return ln_C


def rand_wishart(data_dim, n_states, c=1e+0, by_eye=True):
    '''
    @argv
    data_dim: data_dim
    n_states: n_states
    @return
    W: array(data_dim, data_dim, n_states)
    '''
    from scipy.stats import wishart
    if by_eye:
        from numpy import eye
        W = wishart.rvs(data_dim, eye(data_dim) * c, size=n_states)
    else:
        from numpy import ones
        W = wishart.rvs(data_dim, ones(data_dim) * c, size=n_states)
    if data_dim == 1:
        W = W.reshape(1, 1, n_states)
    else:
        W = W.transpose(1, 2, 0)
    return W


def main_inv_det():
    from scipy.stats import wishart
    from numpy import eye
    data_dim = 4
    n_states = 3
    src = wishart.rvs(data_dim, eye(data_dim), size=n_states)
    print('src:\n', src)
    dst = inv(src)
    print('inv:\n', dst)
    dst = logdet(src)
    print('det:\n', dst)


def main_logsumexp():
    from numpy.random import rand
    data_len = 100
    data_dim = 2
    axis = 1
    src = rand(data_len, data_dim)
    dst = logsumexp(src, axis=axis)
    print(dst)
    print(log(exp(src).sum(axis)))


if __name__ == '__main__':
    main_inv_det()
    main_logsumexp()
