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
from numpy import diag
from numpy.random import multivariate_normal as mvnrnd

from default_hyper_params import ParamMultivariateNormal

cdir = os.path.abspath(os.path.dirname(__file__))
lib_root = os.path.join(cdir, '..')
sys.path.append(lib_root)
from util.calc_util import inv
from util.calc_util import logdet
from util.logger import logger


class MultivariateNormal:
    '''
    mvn = MultivaliateNormal(data_dim, n_states=1)
    mvn.set_params(mean=mu, cov=cov, prec=prec)
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
        self.expt_prec = None
        self.expt_lndet_prec = None
        self.expt_prec_mu = None
        self.expt_mu_prec_mu = None
        if do_set_prm:
            self.set_params()

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
        '''
        param_mvn = ParamMultivariateNormal(self.data_dim, self.n_states)
        mu = kargs.get('mu', None)
        cov = kargs.get('cov', None)
        prec = kargs.get('prec', None)
        if mu is None:
            self.mu = param_mvn.mu
        else:
            self.mu = mu
        if prec is None:
            if cov is None:
                self.cov = param_mvn.cov
                self.prec = self.calc_inv(self.cov)
            else:
                self.cov = cov
                self.prec = self.calc_inv(self.cov)
        else:
            self.prec = prec
            self.cov = self.calc_inv(self.prec)
        self.n_states = self.mu.shape[-1]
        if self.mu.ndim == 1:
            self.data_dim = 1
        else:
            self.data_dim = self.mu.shape[0]
        self.expt_prec = self.prec
        self.expt_lndet_prec = self.calc_lndet(self.prec)
        self.expt_prec_mu = self.calc_prec_mu()
        self.expt_mu_prec_mu = self.calc_mu_prec_mu()

    def get_param_dict(self):
        '''
        prm = mvn.get_param_dict()
        dst: dictionary, {'mu': self.mu, 'cov': self.cov, 'prec': self.prec}
        '''
        dst = {'mu': self.mu, 'cov': self.cov, 'prec': self.prec}
        return dst

    def set_param_dict(self, src):
        '''
        '''
        self._set_param(**src)

    def calc_inv(self, src):
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

    def calc_prec_mu(self):
        if self.mu.ndim == 1:
            if self.cov.ndim == 1:
                pm = self.prec * self.mu
            elif self.cov.ndim == 2:
                pm = einsum('lj,j->l', self.prec, self.mu)
            elif self.cov.ndim == 3:
                pm = einsum('ljk,jk->lk', self.prec, self.mu)
        elif self.mu.ndim == 2:
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
        else:
            self._log_error_not_supported()
            pm = None
        return pm

    def calc_mu_prec_mu(self):
        if self.mu.ndim == 1:
            if self.cov.ndim == 1:
                mpm = self.mu**2 * self.prec
            elif self.cov.ndim == 2:
                mpm = einsum('l,lj,j->', self.mu, self.prec, self.mu)
            elif self.cov.ndim == 3:
                mpm = einsum('lk,ljk,jk->k', self.mu, self.prec, self.mu)
        elif self.mu.ndim == 2:
            if self.cov.ndim == 2:
                mpm = einsum('lk,lj,jk->k', self.mu, self.prec, self.mu)
            elif self.cov.ndim == 3:
                mpm = einsum('lk,ljk,jk->k', self.mu, self.prec, self.mu)
            else:
                self._log_error_not_supported()
        elif self.mu.ndim == 3:
            if self.cov.ndim == 3:
                m = einsum('llk->lk', self.mu)
                mpm = einsum('lk,ljk,jk->k', m, self.prec, m)
            elif self.cov.ndim == 4:
                m = self.mu
                mpm = einsum('ldk,ljdk,jdk->dk', m, self.prec, m)
            else:
                self._log_error_not_supported()
                mpm = None
        else:
            self._log_error_not_supported()
            mpm = None
        return mpm

    def calc_lndet(self, src):
        if src.ndim == 1:
            dst = src
        elif src.ndim == 2:
            dst = logdet(src)
        elif src.ndim == 3:
            dst = logdet(src.transpose(2, 0, 1))
        elif src.ndim == 4:
            dst = logdet(src.transpose(2, 3, 0, 1))
        else:
            self._log_error_not_supported()
            dst = None
        return dst

    def sample(self, data_len=1):
        '''
        @argvs
        data_len: number of samples
        dst: (data_len, data_dim, n_states)
        '''
        dst = zeros((data_len, self.mu.shape[0], self.mu.shape[1]))
        for k in xrange(self.mu.shape[-1]):
            m, c = self.mu[:, k], self.cov[:, :, k]
            dst[:, :, k] = mvnrnd(m, c, size=data_len)
        return dst

    def calc_lkh(self, Y, YY=None):
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

    def _log_error_not_supported(self):
        logger.error('mean and prec of ndim (%d, %d) not supported' %
                     (self.mu.ndim, self.cov.ndim))

    def __str__(self):
        res = '-' * 7
        res += 'type: %s\n' % self.__class__.__name__
        res += 'mu.ndim : %d\n' % self.mu.ndim
        res += 'mu.shape: %s\n' % ','.join(['%d' % x for x in self.mu.shape])
        res += 'mu:\n%s\n' % self._gen_str(self.mu)
        res += ('-' * 7)
        return res

    def _gen_str_list(self, src, fmt='%8.1f', diag_out=False):
        if src.ndim == 1:
            dst = [[fmt % a for a in src]]
        elif src.ndim == 2:
            n = src.shape[-1]
            if diag_out:
                dst = [fmt % a for a in [diag(src)[k] for k in xrange(n)]]
            else:
                dst = [[fmt % a for a in src[:, k]] for k in xrange(n)]
        elif src.ndim == 3:
            n = src.shape[-1]
            dst = [[fmt % a for a in diag(src[:, :, k])] for k in xrange(n)]
        else:
            logger.error('ndim %d is not supported' % src.ndim)
            dst = None
        return dst

    def _gen_str(self, src, fmt='%8.1f'):
        dst = ''
        str_list = self._gen_str_list(src, fmt)
        for item in str_list:
            dst += '%s\n' % ','.join(item)
        return dst


if __name__ == '__main__':
    # ---
    dist_dim = 2
    data_dim = 4
    n_states = 3
    mvn = MultivariateNormal(data_dim, n_states, do_set_prm=True)
    print mvn
    print mvn.mu
    print mvn.cov
    samp_mu = mvn.sample(100)
    print samp_mu
