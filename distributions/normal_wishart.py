#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
normal_wishart.py
'''
import os
import sys

from numpy import newaxis
from numpy import log
from numpy import zeros
from numpy import arange
from numpy import tile
from numpy import einsum
from numpy.random import multivariate_normal as mvnrnd
from scipy.special import digamma
from scipy.stats import wishart

from default_hyper_params import ParamNormalWishart

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))
from util.calc_util import inv
from util.calc_util import logdet
from util.logger import logger


class NormalWishart(object):
    '''
    nw = NormalWishartDistribution(data_dim, n_states=1)
    nw.set_params(mu=mu, beta=beta, nu=nu, W=W)
    ---
    p(Y, mu, R) = p(Y | mu, R) p(mu, R)
    p(mu, R) = Normal(mu | mu, (beta R)^-1) Wishart(R | nu, W)
    Y: data
    mu: random variable from normal distributioin
    R: random variable from whishart distribution
    --- hyperparemeters
    mu: data_dim x n_states
    beta: n_states
    nu: n_states
    W: data_dim x data_dim x n_states
    --- expectations
    expt_prec: <R>, np.array(data_dim, data_dim, n_states)
    expt_lndet_prec: <ln(|R|)>, np.array(n_states)
    expt_prec_mu: <R mu>, np.array(data_dim, n_states)
    expt_mu_prec_mu: <mu^T R mu>, np.array(n_states)
    '''

    def __init__(self, data_dim, n_states=1, do_set_prm=False):
        '''
        nw = NormalWishartDistribution(data_dim, n_states=1)
        @argvs
        data_dim: dimension of the data
        n_states: number of states, defalut 1
        do_set_param: default parameters are set if True, default False
        '''
        self.n_states = n_states
        self.data_dim = data_dim
        self.beta = None
        self.nu = None
        self.mu = None
        self.W = None
        self.inv_W = None
        self._eps = 1e-10
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
        self.beta = None
        self.nu = None
        self.mu = None
        self.W = None
        self.inv_W = None
        self.expt_prec = None
        self.expt_lndet_prec = None
        self.expt_prec_mu = None
        self.expt_mu_prec_mu = None

    def set_params(self, **argvs):
        '''
        set_params(mu=mu, beta=beta, nu=nu, W=W)
        @argvs
        mu: data_dim x n_states
        beta: n_states
        nu: n_states, must be more than data_dim
        W: data_dim x data_dim x n_states
        inv_W: data_dim x data_dim x n_states
        '''
        prm_nw = ParamNormalWishart(self.data_dim, self.n_states)
        # --- beta
        beta = argvs.get('beta', None)
        if beta is not None:
            self.beta = beta
        else:
            if self.beta is None:
                self.beta = prm_nw.beta
        # --- nu
        nu = argvs.get('nu', None)
        if nu is not None:
            self.nu = nu
        else:
            if self.nu is None:
                self.nu = prm_nw.nu
        # --- mu
        mu = argvs.get('mu', None)
        if mu is not None:
            self.mu = mu
        else:
            if self.mu is None:
                self.mu = prm_nw.mu
        # --- W
        W = argvs.get('W', None)
        inv_W = argvs.get('inv_W', None)
        if W is None and inv_W is None:
            if self.W is None:
                self.W = prm_nw.W
                self.inv_W = self._inv(self.W)
        else:
            if W is not None and inv_W is not None:
                self.W = W
                self.inv_W = inv_W
            elif W is None:
                self.inv_W = inv_W
                self.W = self._inv(self.inv_W)
            elif inv_W is None:
                self.W = W
                self.inv_W = self._inv(W)
        self.n_states = self.mu.shape[-1]
        # --- expectations
        if 'expt_prec' in argvs:
            self.expt_prec = argvs['expt_prec']
        else:
            self.expt_prec = self.calc_expt_prec()
        if 'expt_lndet_prec' in argvs:
            self.expt_lndet_prec = argvs['expt_lndet_prec']
        else:
            self.expt_lndet_prec = self.calc_expt_lndet_prec()
        if 'expt_prec_mu' in argvs:
            self.expt_prec_mu = argvs['expt_prec_mu']
        else:
            self.expt_prec_mu = self.calc_expt_prec_mu()
        if 'expt_mu_prec_mu' in argvs:
            self.expt_mu_prec_mu = argvs['expt_mu_prec_mu']
        else:
            self.expt_mu_prec_mu = self.calc_expt_mu_prec_mu()

    def get_param_dict(self):
        '''
        mu: data_dim x n_states
        beta: n_states
        nu: n_states
        W: data_dim x data_dim x n_states
        '''
        prm = {
            'mu': self.mu,
            'beta': self.beta,
            'nu': self.nu,
            'W': self.W,
            'inv_W': self.inv_W,
            'expt_prec': self.expt_prec,
            'expt_ln_det_prec': self.expt_lndet_prec,
            'expt_prec_mu': self.expt_prec_mu,
            'expt_mu_prec_mu': self.expt_mu_prec_mu,
        }
        return prm

    def set_param_dict(self, prm):
        '''
        @argvs
        prm: dictionary with all or some of [mu, beta, nu, W]
        '''
        self.set_params(**prm)

    def sample_mu(self, data_len=1, R=None):
        '''
        mu: (data_len, data_dim, n_states)
        '''
        if R is None:
            R = self.sample_R()[0]
        mu = zeros((data_len, self.data_dim, self.n_states))
        for k in xrange(self.n_states):
            # cov = inv(self.beta[k] * R[:, :, k])
            cov = inv(R[:, :, k])
            try:
                mu[:, :, k] = mvnrnd(self.mu[:, k], cov, size=data_len)
            except RuntimeWarning as e:
                logger.warn('%s %d. not sampled' % (e, mu.shape[0]))
                mu[:, :, k] = self.mu[newaxis, :, k]
            except Exception as e:
                logger.error('%s %d. not sampled.' % (e, mu.shape[0]))
                mu[:, :, k] = self.mu[newaxis, :, k]
        return mu

    def sample_R(self, data_len=1):
        '''
        R: (data_len, data_dim, data_dim, n_states)
        '''
        R = zeros((data_len, self.data_dim, self.data_dim, self.n_states))
        for k in xrange(self.n_states):
            nu = self.nu[k]
            W = self.W[:, :, k]
            try:
                R[:, :, :, k] = wishart.rvs(nu, W, size=data_len)
            except:
                R[:, :, :, k] = tile((nu * W), (data_len, 1, 1))
                logger.warn('whshart.rvs failed set expectations instead')
        return R

    def sample_mu_R(self, data_len=1):
        '''
        R: (data_dim, data_dim, n_states)
        mu: if data_len > 1 (data_len, data_dim, n_states)
            else (data_dim, n_states)
        '''
        R = self.sample_R()[0]
        mu = self.sample_mu(data_len, R=R)
        if data_len == 1:
            mu = mu[0]
        return mu, R

    def calc_lkh(self, Y, YY=None):
        from numpy import pi as np_pi
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

    def calc_expt_prec(self):
        '''
        dst: data_dim x data_dim x n_states
        <R> = \hat{\nu} \hat{W}
        '''
        dst = einsum('k,dek->dek', self.nu, self.W)
        return dst

    def calc_expt_lndet_prec(self):
        '''
        dst: n_states
        <ln|R|> = \sum \phi(\frac{\hat(\nu + 1 - d)}{2})
                  + D * log(2)
                  + \ln(||hat(W)|)
        '''
        nu_d = self.nu[:, newaxis] - arange(self.data_dim)[newaxis, :]
        term1 = digamma((nu_d + 1.0) / 2.0).sum(1)
        term2 = self.data_dim * log(2)
        term3 = logdet(self.W.transpose(2, 0, 1))
        dst = term1 + term2 + term3
        return dst

    def calc_expt_prec_mu(self):
        '''
        dst: data_dim x n_states
        <R\mu> = \hat{\nu} \hat{W} \hat{m}
        '''
        dst = einsum('k,dek,ek->dk', self.nu, self.W, self.mu)
        return dst

    def calc_expt_mu_prec_mu(self):
        '''
        dst: n_states
        <\mu R \mu> = \hat{\nu} \hat{\mu} \hat{W} \hat{\mu}
                      + frac{D}{\hat{\beta}}
                    = \hat{m}^T <R\mu>.
        '''
        tmp1 = einsum('k,dk,dek,ek->k', self.nu, self.mu, self.W, self.mu)
        tmp2 = self.data_dim / self.beta
        dst = tmp1 + tmp2
        return dst

    def _inv(self, src):
        dst = inv(src.transpose(2, 0, 1)).transpose(1, 2, 0)
        return dst


def plotter(nw, figno=1):
    from util.plot_models import PlotModels
    from numpy import atleast_2d
    n_cols = 4
    n_rows = 1
    pm = PlotModels(n_rows, n_cols, figno)
    pm.plot_2d_array((0, 0), nw.mu, title=r'param $\mu$')
    pm.multi_bar((0, 1), atleast_2d(nw.beta), title=r'param $\beta$')
    pm.multi_bar((0, 2), atleast_2d(nw.nu), title=r'param $\nu$')
    pm.tight_layout()
    pm.show()


def main():
    data_dim = 4
    n_states = 3
    nw = NormalWishart(data_dim, n_states, do_set_prm=True)
    print nw.mu
    print nw.beta
    print nw.nu
    print nw.W
    print nw.expt_prec.shape
    print nw.expt_prec_mu.shape
    print nw.expt_mu_prec_mu.shape
    plotter(nw, 1)


if __name__ == '__main__':
    main()
