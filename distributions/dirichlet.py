#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
dirichlet.py
'''
import os
import sys

from numpy import log
from numpy import exp
from numpy import zeros
from numpy import ones
from numpy import newaxis
from numpy.random import dirichlet
from scipy.special import digamma

from .default_hyper_params import ParamDirichlet

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))
from util.calc_util import logsumexp
from util.logger import logger


class Dirichlet(object):
    '''
    Dirichlet()
    '''

    def __init__(self, n_states, len_2d=0, do_set_prm=False):
        '''
        n_states: number of states
        n_dim: 1 is for initial probability, 2 is for transition probability
        '''
        self.n_states = n_states
        self.len_2d = len_2d
        self.n_dim = 1 if len_2d == 0 else 2
        self.alpha = None
        self.ln_alpha = None
        self.expt = None
        self.expt_ln = None
        if do_set_prm:
            self.set_params()

    def clear_params(self):
        self.alpha = None
        self.ln_alpha = None
        self.expt = None
        self.expt_ln = None

    def set_params(self, **keywords):
        alpha = keywords.get('alpha', None)
        ln_alpha = keywords.get('ln_alpha', None)
        if alpha is not None:
            self.alpha = alpha
            self.ln_alpha = log(alpha)
        elif ln_alpha is not None:
            self.ln_alpha = ln_alpha
            self.alpha = exp(ln_alpha)
        else:
            if self.n_dim == 1:
                self.alpha = ParamDirichlet(self.n_states).alpha
            elif self.n_dim == 2:
                self.alpha = ParamDirichlet(self.n_states, self.len_2d).alpha
            else:
                raise NotImplementedError
            self.ln_alpha = log(self.alpha)
        self.n_states = self.alpha.shape[0]
        self.expt = self.calc_expt()
        self.expt_ln = self.calc_expt_ln()

    def get_param_dict(self):
        prm = {
            'alpha': self.alpha,
            'ln_alpha': self.ln_alpha,
        }
        return prm

    def calc_expt(self):
        ndim = self.alpha.ndim
        if ndim == 1:
            # expt = self.alpha / self.alpha.sum()
            expt = exp(self.ln_alpha - logsumexp(self.ln_alpha))
        elif ndim == 2:
            sum_ln_alpha = logsumexp(self.ln_alpha, 0)
            expt = exp(self.ln_alpha - sum_ln_alpha[newaxis, :])
        return expt

    def calc_expt_ln(self):
        ndim = self.alpha.ndim
        if ndim == 1:
            tmp1 = digamma(self.alpha)
            tmp2 = digamma(self.alpha.sum())
            expt_ln = tmp1 - tmp2[newaxis]
        elif ndim == 2:
            tmp1 = digamma(self.alpha)
            tmp2 = digamma(self.alpha.sum(0))
            expt_ln = tmp1 - tmp2[newaxis, :]
        else:
            logger.error('ndim %d is not supported' % ndim)
        return expt_ln

    def sample(self, data_len=1):
        ndim = self.alpha.ndim
        if ndim == 1:
            dst = dirichlet(self.alpha, size=data_len).T
            if data_len == 1:
                dst = dst[:, 0]
        elif ndim == 2:
            dst = zeros((self.n_states, self.len_2d, data_len))
            if self.alpha.shape[0] == self.alpha.shape[1]:
                dst = zeros((self.n_states, self.len_2d, data_len))
                for k in xrange(self.alpha.shape[0]):
                    dst[:, k] = dirichlet(self.alpha[:, k], size=data_len).T
            else:
                for k in xrange(self.len_2d):
                    dst[:, k, :] = dirichlet(self.alpha[:, k], size=data_len).T
            if data_len == 1:
                dst = dst[:, :, 0]
        else:
            logger.error('data dim %d is not supported' % ndim)
            dst = None
        return dst

    def alpha_str_list(self, fmt='%8.2f'):
        return self._gen_str_list(self.alpha, fmt=fmt)

    def ln_alpha_str_list(self):
        return self._gen_str_list(self.ln_alpha)

    def __str__(self):
        res = ('-' * 10)
        res += '\n'
        res += ('class name : %s\n' % self.__class__.__name__)
        res += ('alpha.ndim : %d\n' % self.alpha.ndim)
        res += ('alpha.shape: %s\n' % list(self.alpha.shape))
        res += ('alpha:\n%s' % self._gen_str(self.alpha))
        res += ('expt:\n%s' % self._gen_str(self.expt))
        res += ('-' * 10)
        return res

    def _gen_str_list(self, src, fmt='%8.2f'):
        if src.ndim == 1:
            dst = [[fmt % a for a in src]]
        elif src.ndim == 2:
            dst = []
            for k in xrange(src.shape[0]):
                dst.append([fmt % a for a in src[k]])
        else:
            logger.error('ndim %d is not supported' % src.ndim)
            dst = None
        return dst

    def _gen_str(self, src, fmt='%8.2f'):
        dst = ''
        str_list = self._gen_str_list(src, fmt)
        for item in str_list:
            dst += '%s\n' % ','.join(item)
        return dst

    def plot_alpha_table(self, fmt='%8.2f'):
        import matplotlib.pyplot as plt
        ndim = self.alpha.ndim
        n_states = self.alpha.shape[0]
        rows = ['k:%d' % x for x in xrange(n_states)]
        if ndim == 1:
            cell = zip(*self.alpha_str_list(fmt=fmt))
            cols = None
            colw = ones(len(cell[0])) * 0.6
        elif ndim == 2:
            cell = self.alpha_str_list(fmt=fmt)
            cols = ['prv k:%d' % xx for xx in xrange(n_states)]
            colw = ones(n_states) / float(n_states)
        else:
            logger.error('ndim %d not supported' % ndim)
        plt.table(
            cellText=cell,
            rowLabels=rows,
            colLabels=cols,
            colWidths=colw,
            loc='center',
            cellLoc='center',
            colLoc='center')
        plt.axis('off')


if __name__ == '__main__':
    from IPython import embed
    from numpy.random import randint
    n_states = 5
    alpha = randint(1, 10, size=n_states)
    dd = Dirichlet(n_states)
    dd.set_params(alpha=alpha)
    print(dd)
    embed()
    n_states = 5
    alpha = randint(1, 10, size=(n_states, n_states))
    dd = Dirichlet(n_states, n_states)
    dd.set_params(alpha=alpha)
    print(dd)
    n_states = 5
    data_len = 10
    alpha = randint(1, 10, size=(data_len, n_states))
    dd = Dirichlet(data_len, n_states)
    dd.set_params(alpha=alpha)
    print(dd)
    embed()
