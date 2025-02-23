#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
gamma.py
'''
import os
import sys
from numpy import log
from numpy import ones
from numpy import eye
from numpy import diag
from numpy import einsum
from numpy.random import gamma as gamrand
from scipy.special import digamma

from IPython import embed

CDIR = os.path.abspath(os.path.dirname(__file__))
LIB_ROOT = os.path.join(CDIR, '..')
sys.path.append(LIB_ROOT)
from distributions.default_hyper_params import ParamGamma
from utils.calc_utils import inv
from utils.calc_utils import logsumexp
from utils.logger import logger


class Gamma():

    def __init__(self, a, b):
        self.a_prior = a
        self.b_prior = b
        self.a_post = np.copy(self.a_prior)
        self.b_post = np.copy(self.b_prior)

    def update_posterior(self, data):
        '''
        '''
        self.a_post = self.a_prior + sum(data)
        self.b_post = self.b_prior + len(data)

    def expectation(self):
        expt = self.a_post / self.b_post
        return expt

    def sample(self, size=1):
        a = self.a_post
        scale = 1 / self.b_post
        # dst = stats.gamma(a=a, scale=scale)
        dst = np.random.gamma(a, scale=scale, size=size)
        return dst

    def hyper_parameters(self):
        dst = {
            'a_prior': self.a_prior,
            'b_prior': self.b_prior,
            'a_posterior': self.a_post,
            'b_posterior': self.b_post,
        }
        return dst


class Gamma2():
    '''
    gd = Gamma(n_states)
    '''

    def __init__(self, n_states, data_dim=1, do_set_prm=False):
        '''
        keywords
        a: array(data_dim, n_states)
        b: array(data_dim, n_states)
        expt: array(data_dim, data_dim, n_states)
        self
        a: array(data_dim, n_states)
        b: array(data_dim, n_states)
        expt: array(data_dim, data_dim, n_states)
        inv_expt: array(data_dim, data_dim, n_states)
        lndet: array(n_states)
        '''
        self.data_dim = data_dim
        self.n_states = n_states
        self.a = None
        self.b = None
        self.expt = None
        # self.lndet = None
        # self.inv_expt = None
        # self.diag_expt = None
        if do_set_prm:
            self.set_params()

    def set_params(self, **keywords):
        '''
        argv:
        a: (data_dim, n_states)
        b: (data_dim, n_states)
        self.set_params(a=a, b=b)
        '''
        prm_gam = ParamGamma(self.data_dim, self.n_states)
        self.a = keywords.get('a', prm_gam.a)
        self.b = keywords.get('b', prm_gam.b)
        self.ln_a = log(self.a)
        self.ln_b = log(self.b)
        self.expt = self.calc_expt()
        self.expt_ln = self.calc_expt_ln()
        self.expt_lndet = self.calc_ln_det()
        # self.inv_expt = self.calc_inv(self.expt)
        # self.lndet = self.calc_ln_det()
        # self.diag_expt = self.calc_diag_expt()
        # self.ln_diag_expt = self.calc_ln_diag_expt()

    def get_param_dict(self):
        prm = {'a': self.a, 'b': self.b}
        return prm

    def set_param_dict(self, prm):
        self.set_params(**prm)

    def calc_expt(self):
        tmp = self.a / self.b
        expt = einsum('ij,ik->ijk', eye(tmp.shape[0]), tmp)
        return expt

    def calc_expt_ln(self):
        tmp = self.ln_a - self.ln_b
        expt = einsum('ij,ik->ijk', eye(tmp.shape[0]), tmp)
        return expt

    def calc_ln_det(self):
        dig_a = digamma(self.a)
        lndet = (dig_a - self.ln_b).sum(0)
        return lndet

    def calc_inv(self, src):
        if src.ndim == 2:
            dst = inv(src)
        elif src.ndim == 3:
            dst = inv(src.transpose(2, 0, 1)).transpose(1, 2, 0)
        else:
            logger.error('ndim %d is not supported' % src.ndim)
            dst = None
        return dst

    def calc_diag_expt(self):
        if self.expt.ndim == 2:
            diag_expt = diag(self.expt)
        elif self.expt.ndim == 3:
            diag_expt = einsum('iik->ik', self.expt)
        else:
            logger.error('ndim %d is not supported' % self.expt.ndim)
            diag_expt = None
        return diag_expt

    def calc_ln_diag_expt(self):
        if self.expt_ln.ndim == 2:
            ln_diag_expt = diag(self.expt_ln)
        elif self.expt.ndim == 3:
            ln_diag_expt = einsum('iik->ik', self.expt_ln)
        else:
            logger.error('ndim %d is not supported' % self.expt_ln.ndim)
            ln_diag_expt = None
        return ln_diag_expt

    def samples(self, data_len=1):
        if self.a.ndim == 1:
            dst = gamrand(self.a, 1.0 / self.b)
        elif self.a.ndim == 2:
            dst = gamrand(self.a, 1.0 / self.b)
        elif self.a.ndim == 3:
            logger.error('ndim %d is not supported' % self.expt.ndim)
            dst = None
        else:
            logger.error('ndim %d is not supported' % self.expt.ndim)
            dst = None
        return dst

    def expectations(self):
        return self.expt

    def __str__(self):
        res = '-' * 7
        res += 'type : %s\n' % self.__class__.__name__
        res += 'ndim : %d\n' % self.expt.ndim
        res += 'shape: %s\n' % ','.join(['%d' % x for x in self.expt.shape])
        res += 'expt:\n%s\n' % self._gen_str(self.expt)
        res += ('-' * 7)
        return res

    def expt_str_list(self, fmt='%.0e'):
        return self._gen_str_list(self.diag_expt)

    def diag_expt_str_list(self, fmt='%.0e'):
        return self._gen_str_list(self.diag_expt)

    def _gen_str_list(self, src, fmt='%.0e'):
        if src.ndim == 1:
            dst = [[fmt % a for a in src]]
        elif src.ndim == 2:
            dst = []
            for k in range(src.shape[-1]):
                dst.append([fmt % a for a in src[:, k]])
        elif src.ndim == 3:
            dst = []
            for k in range(src.shape[-1]):
                dst.append([fmt % a for a in diag(src[:, :, k])])
        else:
            logger.error('ndim %d is not supported' % src.ndim)
            dst = None
        return dst

    def _gen_str(self, src, fmt='%.0e'):
        dst = ''
        str_list = self._gen_str_list(src, fmt)
        for item in str_list:
            dst += '%s\n' % ','.join(item)
        return dst


if __name__ == '__main__':
    data_dim = 64
    n_states = 3
    a = ones((data_dim, n_states)) * 1e+3
    b = ones((data_dim, n_states))
    gd = Gamma(a=a, b=b)
    embed(header=__file__)
