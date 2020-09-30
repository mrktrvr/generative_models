#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
multivariate_normal.py
'''

import os
import sys
from numpy import newaxis
from numpy import einsum
from numpy import log
from numpy import pi as np_pi
from numpy import zeros
from numpy.random import multivariate_normal as mvnrnd
from numpy.random import normal as uvnrnd

CDIR = os.path.abspath(os.path.dirname(__file__))

LIB_ROOT = os.path.join(CDIR, '..')
sys.path.append(LIB_ROOT)
from distributions.default_hyper_params import ParamMultivariateNormal
from util.calc_util import inv
from util.calc_util import logdet
from util.logger import logger


class MultivariateNormal(object):
    '''
    Multivariate normal distribution

    mvn = MultivaliateNormal(data_dim, n_states=1)
    mvn.set_params(mean=mu, cov=cov, prec=prec)
    prms = mvn.get_params()
    smp = mvn.sample(data_len=1)
    mvn.clear_params()
    '''
    def __init__(self, data_dim, n_states=1, do_set_prm=False):
        '''
        p(Y|mu, cov) = N(Y|mu, cov)
        mvn = MultivaliateNormal(data_dim, n_states=1)
        mvn.set_params(mu=mu, cov=cov, prec=prec)
        @argvs
        data_dim: data dimension
        n_states: number of states, default 1, this is for mixture model
        do_set_prm: True or False(default), to set default param or not
        @attributes
        mu: np.array(data_dim, n_states), default None
        cov: np.array(data_dim, data_dim, n_states), default None
        prec: np.array(data_dim, data_dim, n_states), default None
        expt_prec: np.array(data_dim, data_dim, n_states), default None
        expt_lndet_prec: np.array( n_states), default None, <ln |prec|>
        expt_prec_mu: np.array(data_dim, n_states), default None, <prec mu>
        expt_mu_prec_mu: np.array(n_states), default None, <mu^T prec mu>
        '''
        self._data_dim = data_dim
        self._n_states = n_states
        self._mu = None
        self._cov = None
        self._prec = None
        self._expt2 = None
        self._expt_prec = None
        self._expt_lndet_prec = None
        self._expt_prec_mu = None
        self._expt_mu_prec_mu = None
        if do_set_prm:
            self.set_params()
            logger.info('Default hyper parameters has been set.')

    def clear_params(self):
        '''
        clear all parameters
        '''
        self._mu = None
        self._cov = None
        self._prec = None
        self._expt_prec = None
        self._expt_lndet_prec = None
        self._expt_prec_mu = None
        self._expt_mu_prec_mu = None

    def set_params(self, **kargs):
        '''
        @kargvs
        mu: (data_dim, n_states) or (n_states)
        cov: (data_dim, data_dim, n_states) or (n_states)
        prec: (data_dim, data_dim, n_states) or (n_states)

        expt2: <x x'> = mu mu' + Sigma
        expt_prec: Sigma^{-1}
        expt_lndet_prec: <ln|Sigma^{-1}|>
        expt_prec_mu: <Sigma^{-1} mu>
        expt_mu_prec_mu: <mu' Sigma^{-1} mu>
        '''
        param_mvn = ParamMultivariateNormal(self._data_dim, self._n_states)
        mu = kargs.get('mu', None)
        cov = kargs.get('cov', None)
        prec = kargs.get('prec', None)
        if mu is None:
            self._mu = param_mvn.mu
        else:
            self._mu = mu
        if prec is None:
            if cov is None:
                self._cov = param_mvn.cov
                self._prec = self._calc_inv(self._cov)
            else:
                self._cov = cov
                self._prec = self._calc_inv(self._cov)
        else:
            self._prec = prec
            self._cov = self._calc_inv(self._prec)
        self._n_states = self._mu.shape[-1]
        if self._mu.ndim == 1:
            self._data_dim = 1
        else:
            self._data_dim = self._mu.shape[0]

        # --- expectations
        self._expt2 = self._calc_expt2()
        self._expt_prec = self._prec
        self._expt_lndet_prec = self._calc_lndet(self._prec)
        self._expt_prec_mu = self._calc_prec_mu()
        self._expt_mu_prec_mu = self._calc_mu_prec_mu()

    def get_params(self):
        '''
        prm = mvn.get_params()
        dst: dictionary, {'mu': self._mu, 'cov': self._cov, 'prec': self._prec}
        '''
        dst = {'mu': self._mu, 'cov': self._cov, 'prec': self._prec}
        return dst

    def sample(self, data_len=1):
        '''
        mvn.sample(data_dim=1)

        Argvs
        data_len: number of samples

        Returns
        dst: (data_len, data_dim, n_states) or (data_len, n_states)
        '''
        dst = None
        if self._mu.ndim == 1:
            if self._cov.ndim == 1:
                n_states = self._mu.shape[-1]
                dst = zeros((data_len, n_states))
                for k, (m, c) in enumerate(zip(self._mu, self._cov)):
                    dst[:, k] = uvnrnd(m, c, size=data_len)
            elif self._cov.ndim == 2:
                dst = mvnrnd(self._mu, self._cov, size=data_len)
            elif self._cov.ndim == 3:
                self.__log_error_not_supported()
        elif self._mu.ndim == 2:
            dst = zeros((data_len, self._mu.shape[0], self._mu.shape[1]))
            if self._cov.ndim == 2:
                for k in range(self._mu.shape[-1]):
                    dst[:, :, k] = mvnrnd(self._mu[:, k],
                                          self._cov,
                                          size=data_len)
            elif self._cov.ndim == 3:
                for k in range(self._mu.shape[-1]):
                    dst[:, :, k] = mvnrnd(self._mu[:, k],
                                          self._cov[:, :, k],
                                          size=data_len)
            else:
                self.__log_error_not_supported()
        elif self._mu.ndim == 3:
            if self._cov.ndim == 3:
                for d in range(self._mu.shape[-2]):
                    for k in range(self._mu.shape[-1]):
                        dst[:, d, k] = mvnrnd(self._mu[d, k],
                                              self._cov[:, :, k],
                                              size=data_len)
            elif self._cov.ndim == 4:
                aug_dim, data_dim, n_states = self._mu.shape
                dst = zeros((data_len, aug_dim, data_dim, n_states))
                for d in range(data_dim):
                    for k in range(n_states):
                        dst[:, :, d, k] = mvnrnd(self._mu[:, d, k],
                                                 self._cov[:, :, d, k],
                                                 size=data_len)
            else:
                self.__log_error_not_supported()
        else:
            self.__log_error_not_supported()
        if data_len == 1:
            dst = dst[0]
        return dst

    def _calc_inv(self, src):
        if src.ndim == 1:
            dst = 1.0 / src
        elif src.ndim == 2:
            dst = inv(src)
        elif src.ndim == 3:
            dst = inv(src.transpose(2, 0, 1)).transpose(1, 2, 0)
        elif src.ndim == 4:
            dst = inv(src.transpose(2, 3, 0, 1)).transpose(2, 3, 0, 1)
        else:
            self._log_error_not_supported()
            dst = None
        return dst

    def _calc_expt2(self):
        '''
        expt2 = <mu mu'> = mu mu' + Sigma
        '''
        if self._mu.ndim == 1:
            if self._cov.ndim == 1:
                mm = einsum('l,l->l', self._mu, self._mu)
            elif self._cov.ndim == 2:
                mm = einsum('l,j->lj', self._mu, self._mu)
            elif self._cov.ndim == 3:
                mm = einsum('l,j->lj', self._mu, self._mu)[:, newaxis]
        elif self._mu.ndim == 2:
            if self._cov.ndim == 2:
                mm = einsum('lk,jk->lj', self._mu, self._mu)
            elif self._cov.ndim == 3:
                mm = einsum('lk,jk->ljk', self._mu, self._mu)
            else:
                mm = None
                self.__log_error_not_supported()
        elif self._mu.ndim == 3:
            if self._cov.ndim == 3:
                mm = einsum('ljk,jlk->ljk', self._mu, self._mu)
            elif self._cov.ndim == 4:
                mm = einsum('ldk,jdk->ljdk', self._mu, self._mu)
            else:
                self.__log_error_not_supported()
                mm = None
        else:
            self.__log_error_not_supported()
            mm = None
        if mm is not None:
            expt2 = self._cov + mm
        else:
            expt2 = None
        return expt2

    def _calc_prec_mu(self):
        if self._mu.ndim == 1:
            if self._cov.ndim == 1:
                pm = self._prec * self._mu
            elif self._cov.ndim == 2:
                pm = einsum('lj,j->l', self._prec, self._mu)
            elif self._cov.ndim == 3:
                pm = einsum('ljk,jk->lk', self._prec, self._mu)
        elif self._mu.ndim == 2:
            if self._cov.ndim == 2:
                pm = einsum('lj,jk->lk', self._prec, self._mu)
            elif self._cov.ndim == 3:
                pm = einsum('ljk,jk->lk', self._prec, self._mu)
            else:
                self.__log_error_not_supported()
        elif self._mu.ndim == 3:
            if self._cov.ndim == 3:
                m = einsum('llk->lk', self._mu)
                pm = einsum('ljk,jk->lk', self._prec, m)
            elif self._cov.ndim == 4:
                pm = einsum('ljdk,jdk->ldk', self._prec, self._mu)
            else:
                self.__log_error_not_supported()
                pm = None
        else:
            self.__log_error_not_supported()
            pm = None
        return pm

    def _calc_mu_prec_mu(self):
        if self._mu.ndim == 1:
            if self._cov.ndim == 1:
                mpm = self._mu**2 * self._prec
            elif self._cov.ndim == 2:
                mpm = einsum('l,lj,j->', self._mu, self._prec, self._mu)
            elif self._cov.ndim == 3:
                mpm = einsum('lk,ljk,jk->k', self._mu, self._prec, self._mu)
        elif self._mu.ndim == 2:
            if self._cov.ndim == 2:
                mpm = einsum('lk,lj,jk->k', self._mu, self._prec, self._mu)
            elif self._cov.ndim == 3:
                mpm = einsum('lk,ljk,jk->k', self._mu, self._prec, self._mu)
            else:
                self.__log_error_not_supported()
        elif self._mu.ndim == 3:
            if self._cov.ndim == 3:
                m = einsum('llk->lk', self._mu)
                mpm = einsum('lk,ljk,jk->k', m, self._prec, m)
            elif self._cov.ndim == 4:
                m = self._mu
                mpm = einsum('ldk,ljdk,jdk->dk', m, self._prec, m)
            else:
                self.__log_error_not_supported()
                mpm = None
        else:
            self.__log_error_not_supported()
            mpm = None
        return mpm

    def _calc_lndet(self, src):
        if src.ndim == 1:
            dst = src
        elif src.ndim == 2:
            dst = logdet(src)
        elif src.ndim == 3:
            dst = logdet(src.transpose(2, 0, 1))
        elif src.ndim == 4:
            dst = logdet(src.transpose(2, 3, 0, 1))
        else:
            self.__log_error_not_supported()
            dst = None
        return dst

    def _calc_lkh(self, Y, YY=None):
        data_dim, data_len = Y.shape
        ln_lkh = zeros((self._n_states, data_len))
        ln_lkh -= 0.5 * data_dim * log(2 * np_pi)
        ln_lkh += 0.5 * self._expt_lndet_prec[:, newaxis]
        if YY is None:
            YY = einsum('dt,et->det', Y, Y)
        ln_lkh -= 0.5 * einsum('dek,det->kt', self._expt_prec, YY)
        ln_lkh += einsum('dt,dk->kt', Y, self._expt_prec_mu)
        ln_lkh -= 0.5 * self._expt_mu_prec_mu[:, newaxis]
        return ln_lkh

    def __log_error_not_supported(self):
        logger.error('mean and prec of ndim (%d, %d) not supported' %
                     (self._mu.ndim, self._cov.ndim))

    def __str__(self):
        res = '\n'.join([
            '-' * 7,
            '%15s: %s' % ('class name', self.__class__.__name__),
            '%15s: %d' % ('data dim', self._data_dim),
            '%15s: %d' % ('num states', self._n_states),
            '%15s: %d' % ('mean  ndim', self._mu.ndim),
            '%15s: %s' % ('mean shape', self._mu.shape),
            '%15s: %d' % ('cov  ndim', self._cov.ndim),
            '%15s: %s' % ('cov shape', self._cov.shape),
            '%15s: %d' % ('precision  ndim', self._cov.ndim),
            '%15s: %s' % ('precision shape', self._cov.shape),
            '-' * 7,
        ])
        return res


if __name__ == '__main__':
    # ---
    dist_dim = 2
    data_dim = 4
    n_states = 3
    data_len = 100
    mvn = MultivariateNormal(data_dim, n_states, do_set_prm=True)
    smp = mvn.sample(data_len)
    print(mvn)
    print(smp.shape)
    print('\n'.join([
        '--- mean',
        '\n'.join([
            '%2d: %s' % (k, ','.join(['%6.2f' % x for x in mvn._mu[:, k]]))
            for k in range(n_states)
        ]),
        '--- cov (diag)',
        '\n'.join([
            'k=%2d:%s' % (k, ','.join(
                ['%6.2f' % mvn._cov[d, d, k] for d in range(data_dim)]))
            for k in range(n_states)
        ]),
        '--- samples',
        '\n'.join([
            't:%4d - %s' % (t, ', '.join([
                'k:%2d [%s]' %
                (k, ','.join(['%6.2f' % smp_t[d, k] for d in range(data_dim)]))
                for k in range(n_states)
            ])) for t, smp_t in enumerate(smp)
        ]),
    ]))
