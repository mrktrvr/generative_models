#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from numpy import nan
from numpy import newaxis
from numpy import arange
from numpy import zeros
from numpy import ones
from numpy import log

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))
from utils.calc_utils import logmatprod
from utils.calc_utils import logsumexp


class ForwardBackward(object):
    '''
    ln_expt, ln_expt2, ln_lkh = ForwardBackward()(ln_obs_lkh, ln_pi, ln_trans)
    @argvs
    ln_obs_lkh: array(n_states, data_len)
    ln_pi: array(n_states)
    ln_A: array(n_states, n_states)
    @attributes
    ln_lkh: float
    ln_c: array(data_len)
    ln_fwd: array(n_states, data_len)
    ln_bwd: array(n_states, data_len)
    ln_expt: array(n_states, data_len)
    ln_expt2: array(n_states, n_states, data_len - 1)
    '''

    def __init__(self):
        '''
        '''
        pass

    @staticmethod
    def _calc_fwd(ln_obs_lkh, ln_pi, ln_trans):
        '''
        fwd_0 = p(s_0) p(y_0 | s_0)
        fwd_t = p(y_t | s_t) sum(fwd(s_{t-1}) p(s_t |p(s_{t-1})))
        '''
        n_states, data_len = ln_obs_lkh.shape
        ln_fwd = ones((n_states, data_len)) * nan
        ln_c = zeros((data_len)) * nan
        for t in range(0, data_len):
            if t == 0:
                tmp = ln_obs_lkh[:, t] + ln_pi
            else:
                lmp = logmatprod(ln_fwd[:, t - 1], ln_trans)
                tmp = ln_obs_lkh[:, t] + lmp
            ln_fwd[:, t] = tmp
            ln_c[t] = logsumexp(tmp)
            ln_fwd[:, t] -= ln_c[t]
        return ln_fwd, ln_c

    @staticmethod
    def _calc_bwd(ln_obs_lkh, ln_trans, ln_c):
        '''
        bwd = sum( bwd(s_{t+1}) p(y_{t+1}|s_{t+1}) p(s_{t+1}| s_t))
        '''
        n_states, data_len = ln_obs_lkh.shape
        ln_bwd = ones((n_states, data_len)) * nan
        ln_bwd[:, -1] = 0
        for t in range(data_len - 1, 0, -1):
            tmp = ln_bwd[:, t] + ln_obs_lkh[:, t]
            ln_bwd[:, t - 1] = logmatprod(tmp, ln_trans.T)
            ln_bwd[:, t - 1] -= ln_c[t]
        return ln_bwd

    @staticmethod
    def _calc_ln_expt(ln_fwd, ln_bwd):
        '''
        ln_expt(s_t) = alpha(s_t) beta(s_t)
        @return
        n_expt: (self.n_states, self.data_len)
        '''
        ln_expt = ln_fwd + ln_bwd
        return ln_expt

    @staticmethod
    def _calc_ln_expt2(ln_obs_lkh, ln_trans, ln_fwd, ln_bwd, ln_c):
        '''
        log joint posterior distribution P(S_t-1, S_t | Y)
        ln_expt2 = fwd(s_{t-1}) p(y_t|s_t) p(s_t|s_{t-1}) bwd(s_t) / c
        @return
        ln_expt2: (self.n_states, self.n_states, self.data_len - 1)
        '''
        ln_expt2 = ln_fwd[:, newaxis, :-1] + ln_bwd[newaxis, :, 1:]
        ln_expt2 += ln_obs_lkh[newaxis, :, 1:]
        ln_expt2 += ln_trans[:, :, newaxis]
        ln_expt2 -= ln_c[newaxis, newaxis, 1:]
        return ln_expt2

    def __call__(self, ln_obs_lkh, ln_pi, ln_trans):
        '''
        ln_expt: array(n_states, data_len)
        ln_expt2: array(n_states, n_states, data_len - 1)
        ln_lkh: float
        '''
        ln_fwd, ln_c = self._calc_fwd(ln_obs_lkh, ln_pi, ln_trans)
        ln_bwd = self._calc_bwd(ln_obs_lkh, ln_trans, ln_c)
        ln_expt = self._calc_ln_expt(ln_fwd, ln_bwd)
        ln_expt2 = self._calc_ln_expt2(ln_obs_lkh, ln_trans, ln_fwd, ln_bwd,
                                       ln_c)
        ln_lkh = ln_c.sum()
        return ln_expt, ln_expt2, ln_lkh


def gen_data(data_len, n_states):
    from numpy.random import dirichlet
    ln_obs_lkh = arange(data_len * n_states).reshape(n_states, data_len)
    ln_pi = log(dirichlet(ones(n_states)))
    ln_A = log(dirichlet(ones(n_states), size=n_states))
    return ln_obs_lkh, ln_pi, ln_A


def main():
    import time
    data_len = 1000
    n_states = 20
    # --- data
    ln_obs_lkh, ln_pi, ln_A = gen_data(data_len, n_states)
    # --- forward backward
    fb_alg = ForwardBackward(ln_obs_lkh, ln_pi, ln_A)
    t_bgn = time.time()
    ln_expt, ln_expt2, ln_lkh = fb_alg()
    t_end = time.time()
    print('data_len : %d' % data_len)
    print('n_states : %d' % n_states)
    print('ln_expt  : shape = %s' % list(ln_expt.shape))
    print('ln_expt2 : shape = %s' % list(ln_expt2.shape))
    print('ln_lkh   : %f' % ln_lkh)
    print('calc time: %.3f' % (t_end - t_bgn))


if __name__ == '__main__':
    main()
