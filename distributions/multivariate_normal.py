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
from numpy import atleast_2d
from numpy import zeros
from numpy.random import multivariate_normal as mvnrnd

CDIR = os.path.abspath(os.path.dirname(__file__))

LIB_ROOT = os.path.join(CDIR, '..')
sys.path.append(LIB_ROOT)
from distributions.default_hyper_params import MultivariateNormalParams
from util.calc_util import inv
from util.calc_util import calc_lndet
from util.logger import logger


class MultivariateNormal(object):
    '''
    Multivariate normal distribution

    mvn = MultivaliateNormal(data_dim, n_states=1)
    mvn.set_params(mean=mu, cov=cov, prec=prec)
    prms = mvn.get_params()
    smp = mvn.samples(data_len=1)
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
        self.data_dim = data_dim
        self.n_states = n_states
        self.mu = None
        self.cov = None
        self.prec = None
        self.expt2 = None
        self.expt_prec = None
        self.expt_lndet_prec = None
        self.expt_prec_mu = None
        self.expt_mu_prec_mu = None
        if do_set_prm:
            self.set_params()
            logger.info('Default hyper parameters has been set.')

    def clear_params(self):
        '''
        clear all parameters
        '''
        self.mu = None
        self.cov = None
        self.prec = None
        self.expt_prec = None
        self.expt_lndet_prec = None
        self.expt_prec_mu = None
        self.expt_mu_prec_mu = None

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
        mvn_prms = MultivariateNormalParams(self.data_dim, self.n_states)
        mu = kargs.get('mu', None)
        cov = kargs.get('cov', None)
        prec = kargs.get('prec', None)
        # --- mu
        if mu is None:
            self.mu = mvn_prms.mu
        else:
            self.mu = mu
        if self.mu.ndim == 1:
            self.mu = atleast_2d(self.mu)

        # --- precision and covariance
        if prec is None:
            if cov is None:
                self.cov = mvn_prms.cov
                if self.cov.ndim == 1:
                    self.cov = atleast_2d(self.cova)
                self.prec = self._calc_inv(self.cov)
            else:
                self.cov = cov
                if self.cov.ndim == 1:
                    self.cov = atleast_2d(self.cova)
                self.prec = self._calc_inv(self.cov)
        else:
            self.prec = prec
            if self.prec.ndim == 1:
                self.prec = atleast_2d(self.prec)
            self.cov = self._calc_inv(self.prec)

        # --- data size
        self.data_dim = self.mu.shape[0]
        self.n_states = self.mu.shape[-1]

        # --- expectations
        self.expt2 = self._calc_expt2()
        self.expt_prec = self.prec
        self.expt_lndet_prec = calc_lndet(self.prec)
        self.expt_prec_mu = self._calc_prec_mu()
        self.expt_mu_prec_mu = self._calc_mu_prec_mu()

    def get_params(self):
        '''
        prm = mvn.get_params()
        dst: dictionary, {'mu': self.mu, 'cov': self.cov, 'prec': self.prec}
        '''
        dst = {'mu': self.mu, 'cov': self.cov, 'prec': self.prec}
        return dst

    def samples(self, data_len=1):
        '''
        mvn.samples(data_dim=1)

        Argvs
        data_len: number of samples

        Returns
        dst: (data_len, data_dim, n_states) or (data_len, n_states)
        '''
        dst = None
        if self.mu.ndim == 2:
            dst = zeros((data_len, self.mu.shape[0], self.mu.shape[1]))
            if self.cov.ndim == 2:
                for k in range(self.mu.shape[-1]):
                    dst[:, :, k] = mvnrnd(
                        self.mu[:, k], self.cov, size=data_len)
            elif self.cov.ndim == 3:
                for k in range(self.mu.shape[-1]):
                    dst[:, :, k] = mvnrnd(
                        self.mu[:, k], self.cov[:, :, k], size=data_len)
            else:
                self._log_error_not_supported()
        elif self.mu.ndim == 3:
            if self.cov.ndim == 3:
                for d in range(self.mu.shape[-2]):
                    for k in range(self.mu.shape[-1]):
                        dst[:, d, k] = mvnrnd(
                            self.mu[d, k], self.cov[:, :, k], size=data_len)
            elif self.cov.ndim == 4:
                aug_dim, data_dim, n_states = self.mu.shape
                dst = zeros((data_len, aug_dim, data_dim, n_states))
                for d in range(data_dim):
                    for k in range(n_states):
                        dst[:, :, d, k] = mvnrnd(
                            self.mu[:, d, k],
                            self.cov[:, :, d, k],
                            size=data_len)
            else:
                self.__log_error_not_supported()
        elif self.mu.ndim == 1:
            raise Exception('self.mu.ndim must be >= 1')
        else:
            self.__log_error_not_supported()
        if data_len == 1:
            dst = dst[0]
        return dst

    def expectations(self):
        return self.mu, self.expt_prec

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
        if self.mu.ndim == 2:
            if self.cov.ndim == 2:
                mm = einsum('lk,jk->lj', self.mu, self.mu)
            elif self.cov.ndim == 3:
                mm = einsum('lk,jk->ljk', self.mu, self.mu)
            else:
                mm = None
                self._log_error_not_supported()
        elif self.mu.ndim == 3:
            if self.cov.ndim == 3:
                mm = einsum('ljk,jlk->ljk', self.mu, self.mu)
            elif self.cov.ndim == 4:
                mm = einsum('ldk,jdk->ljdk', self.mu, self.mu)
            else:
                self._log_error_not_supported()
                mm = None
        elif self.mu.ndim == 1:
            raise Exception('self.mu.ndim must be >= 1')
        else:
            self.__log_error_not_supported()
            mm = None
        if mm is not None:
            expt2 = self.cov + mm
        else:
            expt2 = None
        return expt2

    def _calc_prec_mu(self):
        if self.mu.ndim == 2:
            if self.cov.ndim == 2:
                pm = einsum('lj,jk->lk', self.prec, self.mu)
            elif self.cov.ndim == 3:
                pm = einsum('ljk,jk->lk', self.prec, self.mu)
            else:
                self._log_error_not_supported()
        elif self.mu.ndim == 3:
            if self.cov.ndim == 3:
                m = einsum('llk->lk', self.mu)
                pm = einsum('ljk,jk->lk', self.prec, m)
            elif self.cov.ndim == 4:
                pm = einsum('ljdk,jdk->ldk', self.prec, self.mu)
            else:
                self._log_error_not_supported()
                pm = None
        elif self.mu.ndim == 1:
            raise Exception('self.mu.ndim must be >= 1')
        else:
            self.__log_error_not_supported()
            pm = None
        return pm

    def _calc_mu_prec_mu(self):
        if self.mu.ndim == 2:
            if self.cov.ndim == 2:
                mpm = einsum('lk,lj,jk->k', self.mu, self.prec, self.mu)
            elif self.cov.ndim == 3:
                mpm = einsum('lk,ljk,jk->k', self.mu, self.prec, self.mu)
            else:
                self.__log_error_not_supported()
        elif self.mu.ndim == 3:
            if self.cov.ndim == 3:
                m = einsum('llk->lk', self.mu)
                mpm = einsum('lk,ljk,jk->k', m, self.prec, m)
            elif self.cov.ndim == 4:
                m = self.mu
                mpm = einsum('ldk,ljdk,jdk->dk', m, self.prec, m)
            else:
                self.__log_error_not_supported()
                mpm = None
        elif self.mu.ndim == 1:
            raise Exception('self.mu.ndim must be >= 1')
        else:
            self.__log_error_not_supported()
            mpm = None
        return mpm

    def _calc_lkh(self, Y, YY=None):
        data_dim, data_len = Y.shape
        ln_lkh = zeros((self.n_states, data_len))
        ln_lkh -= 0.5 * data_dim * log(2 * np_pi)
        ln_lkh += 0.5 * self.expt_lndet_prec[:, newaxis]
        if YY is None:
            YY = einsum('dt,et->det', Y, Y)
        ln_lkh -= 0.5 * einsum('dek,det->kt', self.expt_prec, YY)
        ln_lkh += einsum('dt,dk->kt', Y, self.expt_prec_mu)
        ln_lkh -= 0.5 * self.expt_mu_prec_mu[:, newaxis]
        return ln_lkh

    def __log_error_not_supported(self):
        logger.error('mean and prec of ndim (%d, %d) not supported' %
                     (self.mu.ndim, self.cov.ndim))

    def __str__(self):
        res = '\n'.join([
            '-' * 7,
            '%15s: %s' % ('class name', self.__class__.__name__),
            '%15s: %d' % ('data dim', self.data_dim),
            '%15s: %d' % ('num states', self.n_states),
            '%15s: %d' % ('mean  ndim', self.mu.ndim),
            '%15s: %s' % ('mean shape', self.mu.shape),
            '%15s: %d' % ('cov  ndim', self.cov.ndim),
            '%15s: %s' % ('cov shape', self.cov.shape),
            '%15s: %d' % ('precision  ndim', self.cov.ndim),
            '%15s: %s' % ('precision shape', self.cov.shape),
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
    smp = mvn.samples(data_len)
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
            'k=%2d:%s' %
            (k,
             ','.join(['%6.2f' % mvn._cov[d, d, k] for d in range(data_dim)]))
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
