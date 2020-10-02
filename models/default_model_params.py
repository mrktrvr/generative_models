#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
fa.py
'''
import os
import sys

from numpy import nan
from numpy import append
from numpy import zeros
from numpy import ones
from numpy import tile
from numpy import eye
from numpy.random import randn

CDIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CDIR, '..'))
from util.calc_util import inv


class DefaultFaParams(object):
    @classmethod
    def Lamb(cls, fa_dim, data_dim, n_states, **keywords):
        '''
        DefaultPriors().Lamb(fa_dim, data_dim, n_states)
        lm: array(aug_dim, data_dim, n_states)
        lc: array(aug_dim, aug_dim, data_dim, n_states)
        '''
        aug_dim = fa_dim + 1
        lm = ones((aug_dim, data_dim, n_states)) * nan
        lm[:-1, :, :] = randn(fa_dim, data_dim, n_states) * 3
        lm[-1, :, :] = randn(1, n_states) * 2
        # --- cov
        _cf = keywords.get('cov', 1e+0)
        tmp = tile(eye(aug_dim) * _cf, (data_dim, n_states, 1, 1))
        lc = tmp.transpose(2, 3, 0, 1)
        return lm, lc

    @classmethod
    def R(cls, data_dim, n_states, **keywords):
        _a = keywords.get('a', 1e+0)
        _b = keywords.get('b', 1e+1)
        a = ones((data_dim, n_states)) * _a
        b = ones((data_dim, n_states)) * _b
        return a, b

    @classmethod
    def Z(cls, fa_dim):
        '''
        DefaultPriors().Z(fa_dim)
        @argvs
        fa_dim: dimention of factor analysis
        @output
        z_mean: np.array(aug_dim)
        z_cov: np.array(aug_dim, aug_dim)
        z_prc: np.array(aug_dim, aug_dim)
        '''
        aug_dim = fa_dim + 1
        # --- Z
        fa_dim = int(aug_dim - 1)
        eps = 1e-8
        # eps = get_eps()
        # eps = get_eps() * 1e+10
        z_mean = append(zeros(fa_dim), [1])
        z_cov = eye(aug_dim)
        z_cov[-1, -1] = eps
        z_prc = inv(z_cov)
        return z_mean, z_cov, z_prc


class DefaultLdaParams(object):
    @classmethod
    def Pi(cls, n_states, n_cat):
        '''
        DefaultPriors().A(n_states, len_2d)
        alpha_pi: (n_states, n_cat)
        '''
        alpha_pi = ones((n_states, n_cat)) * 10
        return alpha_pi

    @classmethod
    def Phi(clf, n_cat):
        alpha_phi = ones(n_cat)
        return alpha_phi
