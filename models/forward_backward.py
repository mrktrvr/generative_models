#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from numpy import nan
from numpy import newaxis
from numpy import arange
from numpy import zeros
from numpy import ones
from numpy import eye
from numpy import tile
from numpy import log
from numpy import sum as nsum

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))
from util.calc_util import logmatprod
from util.calc_util import logsumexp


class ForwardBackward(object):
    '''
    '''

    def __init__(self, ln_obs_lkh, ln_pi, ln_trans):
        self.ln_obs_lkh = ln_obs_lkh
        self.ln_pi = ln_pi
        self.ln_trans = ln_trans
        self.data_len, self.n_states = ln_obs_lkh.shape
        self.ln_c = None
        self.ln_fwd = None
        self.ln_bwd = None
        self.ln_expt = None
        self.ln_expt2 = None

    def _prepare_containers(self):
        self.ln_c = zeros((self.data_len))
        self.ln_fwd = ones((self.data_len, self.n_states)) * nan
        self.ln_bwd = ones((self.data_len, self.n_states)) * nan
        self.ln_expt = ones((self.n_states, self.data_len)) * nan
        self.ln_expt2 = ones(
            (self.n_states, self.n_states, self.data_len - 1)) * nan

    def _calc_fwd(self):
        '''
        fwd_0 = p(s_0) p(y_0 | s_0)
        fwd_t = p(y_t | s_t) sum(fwd(s_{t-1}) p(s_t |p(s_{t-1})))
        '''
        for t in xrange(0, self.data_len):
            if t == 0:
                tmp = self.ln_obs_lkh[t] + self.ln_pi
            else:
                lmp = logmatprod(self.ln_fwd[t - 1], self.ln_trans)
                tmp = self.ln_obs_lkh[t] + lmp
            self.ln_fwd[t] = tmp
            self.ln_c[t] = logsumexp(self.ln_fwd[t])
            self.ln_fwd[t] -= self.ln_c[t]

    def _calc_bwd(self):
        '''
        bwd = sum( bwd(s_{t+1}) p(y_{t+1}|s_{t+1}) p(s_{t+1}| s_t))
        '''
        self.ln_bwd[-1] = 0
        for t in xrange(self.data_len - 1, 0, -1):
            tmp = self.ln_bwd[t] + self.ln_obs_lkh[t]
            self.ln_bwd[t - 1] = logmatprod(tmp, self.ln_trans.T)
            self.ln_bwd[t - 1] -= self.ln_c[t]

    def _calc_ln_expt(self):
        '''
        ln_expt(s_t) = alpha(s_t) beta(s_t)
        '''
        tmp = self.ln_fwd + self.ln_bwd
        self.ln_expt[:, :] = tmp.T

    def _calc_ln_expt2(self):
        '''
        log joint posterior distribution P(S_t-1, S_t | Y)
        ln_expt2 = fwd(s_{t-1}) p(y_t|s_t) p(s_t|s_{t-1}) bwd(s_t) / c
        '''
        for t in xrange(self.data_len - 1):
            a = tile(self.ln_fwd[newaxis, t].T, (1, self.n_states))
            b = tile(self.ln_bwd[newaxis, t + 1], (self.n_states, 1))
            e = tile(self.ln_obs_lkh[newaxis, t + 1], (self.n_states, 1))
            tmp = a + e + self.ln_trans + b - self.ln_c[t + 1]
            self.ln_expt2[:, :, t] = tmp

    def __call__(self):
        '''
        '''
        self._prepare_containers()
        self._calc_fwd()
        self._calc_bwd()
        self._calc_ln_expt()
        self._calc_ln_expt2()
        ln_expt = self.ln_expt.T
        ln_expt2 = self.ln_expt2
        ln_lkh = nsum(self.ln_c)
        return ln_expt, ln_expt2, ln_lkh


def gen_data(data_len, n_states):
    def _prepare_ln_pi(n_states):
        '''
        log pi
        '''
        pi = ones(n_states) * 0.1
        pi[0] = 1
        pi /= nsum(pi)
        ln_pi = log(pi)
        return ln_pi

    def _prepare_ln_A(n_states):
        '''
        log transition
        '''
        if False:
            trans = ones((n_states, n_states)) * 0.1
            trans += eye(n_states)
            trans /= nsum(trans, 0)
            ln_A = log(trans)
        else:
            from numpy.random import dirichlet
            ln_A = log(dirichlet(ones(n_states), size=n_states))
        return ln_A

    ln_obs_lkh = arange(data_len * n_states).reshape(data_len, n_states)
    ln_pi = _prepare_ln_pi(n_states)
    ln_A = _prepare_ln_A(n_states)
    return ln_obs_lkh, ln_pi, ln_A


def main():
    import time
    data_len = 3600
    n_states = 20
    # --- data
    ln_obs_lkh, ln_pi, ln_A = gen_data(data_len, n_states)
    # --- forward backward
    fb_alg = ForwardBackward(ln_obs_lkh, ln_pi, ln_A)
    t_bgn = time.time()
    ln_gamma2, ln_xi2, ln_lkh2 = fb_alg()
    t_end = time.time()
    print 'data_len: %d' % data_len
    print 'n_states: %d' % n_states
    print 'time    : %.3f' % (t_end - t_bgn)


if __name__ == '__main__':
    main()
