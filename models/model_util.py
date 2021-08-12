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
from python_utilities.utils.logger import logger


class CheckTools(object):
    decimals = 5
    min_itr = 10

    @classmethod
    def log_info_update_itr(cls, max_itr, itr, msg='', interval_digit=1):
        len_digit = len('%d' % max_itr)
        if len_digit - interval_digit < 0:
            interval_digit = 0
        interval = 10**interval_digit
        if itr % interval == 0 or itr == (max_itr - 1):
            logger.info('%s update :%5d / %5d (interval %d)' %
                        (msg, itr + 1, max_itr, interval))
        logger.debug('%s update :%5d / %5d' % (msg, itr, max_itr))

    @classmethod
    def check_vb_increase(cls, vbs, i):
        dst = False
        if i < 1:
            dst = True
        else:
            vb_prv = nround(vbs[i - 1], decimals=cls.decimals)
            vb_cur = nround(vbs[i], decimals=cls.decimals)
            vb_diff = vb_cur - vb_prv
            if vb_diff < 0:
                logger.error(' '.join([
                    'vb decreased.',
                    'diff: %.10f' % vb_diff,
                    'iter %3d: %.10f' % (i, vb_cur),
                    'iter %3d: %.10f' % (i - 1, vb_prv),
                ]))
                dst = False
            if vb_cur == vb_prv:
                dst = True
            else:
                dst = True
        return dst

    @classmethod
    def is_conversed(cls, vbs, i, th):
        dst = False
        if i >= cls.min_itr:
            vb_prv = nround(vbs[i - 1], decimals=cls.decimals)
            vb_cur = nround(vbs[i], decimals=cls.decimals)
            vb_diff = vb_cur - vb_prv
            dst = True if vb_diff < th else False
        if dst:
            logger.info(' '.join([
                'Conversed.',
                'iteration at %d.' % i,
                'VB diff %f < %f' % (vb_diff, th),
                '%3d: %f' % (i - 1, vbs[i - 1]),
                '%3d: %f' % (i, vbs[i]),
            ]))

        return dst


def init_expt(data_len, n_states, obs=None, mode='random'):
    '''
    mode: random or kmeans
    '''
    expt = ones((n_states, data_len)) / n_states
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
    elif mode == 'random':
        logger.info('mode=random')
        alpha_pi = ones(n_states)
        expt = dirichlet(alpha_pi, size=data_len).T
    else:
        logger.info('mode=flat (%s not supported)' % mode)
    return expt


if __name__ == '__main__':
    pass
