#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
default_hyper_params.py
'''
import os
import sys

from numpy import eye
from numpy import ones
from numpy import zeros
from numpy import tile
from numpy.random import randn

cdir = os.path.abspath(os.path.dirname(__file__))
lib_root = os.path.join(cdir, '..')
sys.path.append(lib_root)


class ParamMultivariateNormal(object):
    def __init__(self, data_dim, n_states=1, **kargs):
        '''
        @argvs
        data_dim: int, data dimension
        n_states: int, default 1, parameter for mixture model
        kargs:
        @attribute
        mu: np.array(data_dim, n_states)
        cov: np.array(data_dim, data_dim, n_states)
        prec: np.array(data_dim, data_dim, n_states)
        '''
        self.data_dim = data_dim
        self.n_states = n_states
        self.mu = None
        self.cov = None
        self.prec = None
        self._gen_mu()
        self._gen_cov()

    def _gen_mu(self):
        self.mu = randn(self.data_dim, self.n_states)

    def _gen_cov(self):
        # c = 1e+2
        c = 1e+0
        e = eye(self.data_dim)
        self.cov = tile(e, (self.n_states, 1, 1)).transpose(1, 2, 0) * c


class ParamNormalWishart(object):
    def __init__(self, data_dim, n_states):
        '''
        @argvs
        data_dim: int, data dimension
        n_states: int, default 1, parameter for mixture model
        kargs:
        @attribute
        mu: np.array(data_dim, n_states)
        beta: np.array(n_states)
        nu: np.array(n_states)
        W: np.array(data_dim, data_dim, n_states)
        '''
        self.data_dim = data_dim
        self.n_states = n_states
        self.mu = None
        self.beta = None
        self.nu = None
        self.W = None
        self._gen_mu()
        self._gen_beta()
        self._gen_nu()
        self._gen_W()

    def _gen_mu(self):
        self.mu = zeros((self.data_dim, self.n_states))
        # self.mu = randn(self.data_dim, self.n_states)

    def _gen_beta(self):
        self.beta = ones(self.n_states) * 1e+0

    def _gen_W(self, c=1e-1, by_eye=True):
        from util.calc_util import rand_wishart
        self.W = rand_wishart(self.data_dim, self.n_states, c, by_eye)
        self.W = self.W.transpose(1, 2, 0)

    def _gen_nu(self):
        self.nu = ones((self.n_states)) * self.data_dim * 1e+1


class ParamDirichlet(object):
    def __init__(self, n_states, len_2d=-1):
        self.n_states = n_states
        self.len_2d = len_2d
        self.alpha = None
        self._gen_alpha()

    def _gen_alpha(self, c=2e+0):
        if self.len_2d == -1:
            self.alpha = ones(self.n_states) * c
        else:
            self.alpha = ones((self.n_states, self.len_2d)) * c


class ParamGamma(object):
    def __init__(self, data_dim, n_states):
        self.data_dim = data_dim
        self.n_states = n_states
        self.a = None
        self.b = None

    def _gen_gam(self, c=1e+0):
        a = ones((self.data_dim, self.n_states)) * c
        b = ones((self.data_dim, self.n_states)) * c
        return a, b


def main_multivariate_normal():
    data_dim = 4
    n_states = 3
    prm_mvn = ParamMultivariateNormal(data_dim, n_states)
    print prm_mvn.mu.shape
    print prm_mvn.mu
    print prm_mvn.cov.shape


def main_normal_wishart():
    data_dim = 4
    n_states = 3
    prm_nw = ParamNormalWishart(data_dim, n_states)
    print prm_nw.mu.shape
    print prm_nw.beta.shape
    print prm_nw.nu.shape
    print prm_nw.W.shape


if __name__ == '__main__':
    main_multivariate_normal()
    main_normal_wishart()
