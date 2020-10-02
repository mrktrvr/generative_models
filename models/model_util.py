#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_util.py
'''
import os
import sys

from numpy import zeros
from numpy import ones
from numpy import round as nround
from numpy.random import dirichlet

from sklearn.cluster import KMeans

CDIR = os.path.abspath(os.path.dirname(__file__))

LIB_ROOT = os.path.join(CDIR, '..')
sys.path.append(LIB_ROOT)
from util.logger import logger


class CheckTools(object):
    @classmethod
    def log_info_update_itr(cls, max_itr, itr, msg='', interval_digit=1):
        len_digit = len('%d' % max_itr)
        if len_digit - interval_digit < 0:
            interval_digit = 0
        interval = 10**interval_digit
        if itr % interval == 0 or itr == (max_itr - 1):
            logger.info('%s update :%5d / %5d (interval %d)' %
                        (msg, itr, max_itr, interval))
        logger.debug('%s update :%5d / %5d' % (msg, itr, max_itr))

    @classmethod
    def check_vb_increase(cls, vbs, i, decimals=0):
        if i < 1:
            return True
        vb_prv = nround(vbs[i - 1], decimals=decimals)
        vb_cur = nround(vbs[i], decimals=decimals)
        if vb_prv > vb_cur:
            logger.error('vb decreased. iter:%d, %.10f(%.10f->%.10f)' %
                         (i, vbs[i - 1] - vbs[i], vbs[i - 1], vbs[i]))
            return False
        else:
            return True


def init_expt(data_len, n_states, obs=None, mode='random'):
    alpha_pi = ones(n_states)
    expt = dirichlet(alpha_pi, size=data_len).T
    if mode == 'kmeans' and obs is not None:
        logger.info('mode=kmeans')
        if data_len != obs.shape[-1]:
            logger.warning('data_len is different from obs.shape[-1]')
        km = KMeans(n_clusters=n_states)
        km.fit(obs.T)
        eps = 1e-2
        expt_by_km = zeros((n_states, obs.shape[-1]))
        for k in range(n_states):
            expt_by_km[k, km.labels_ == k] = 1 - (eps * (n_states - 1))
        expt_by_km[expt_by_km == 0] = eps
        expt = (expt + expt_by_km) / 2.0
    else:
        logger.info('mode=random')
    return expt


if __name__ == '__main__':
    pass
