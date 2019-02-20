#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_util.py
'''
import os
import sys

from numpy import newaxis
from numpy import zeros
from numpy import ones
from numpy import round as nround
from numpy.random import dirichlet
from sklearn.cluster import KMeans

cdir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(cdir, '..'))
from util.logger import logger


class ModelUtil(object):
    eps = 1e-10
    off_th = 15
    small_pow_app_th = 200

    @classmethod
    def init_expt_s(cls, data_len, n_states, init_mode, **keywords):
        '''
        data_len:
        keywords:
            Y: (data_dim, data_len)
            S: (data_len)
            qPi: qPi object
        expt: (n_states, data_lem)
        '''
        Y = keywords.get('Y', None)
        S = keywords.get('S', None)
        logger.debug('init_mode: %s' % init_mode)
        if S is not None:
            if S.ndim == 1:
                expt = ones((n_states, data_len)) * cls.eps
                v = 1 - cls.eps * (n_states - 1)
                for k in xrange(n_states):
                    expt[k, S == k] = v
            else:
                expt = S
        elif Y is not None:
            if init_mode == 'kmeans':
                expt = cls.expt_by_kmeans(Y.T, n_states)
            elif init_mode == 'random':
                qPi = keywords.get('qPi', None)
                expt = cls.expt_by_dirichlet(qPi, data_len, n_states)
        else:
            if init_mode == 'uniform':
                expt = ones((n_states, data_len), dtype=float) / n_states
            else:
                expt = cls.expt_by_dirichlet(None, data_len, n_states)
        return expt

    @classmethod
    def expt_by_dirichlet(cls, qPi, data_len, n_states):
        if qPi is None:
            expt = dirichlet(ones(n_states), data_len).T
        else:
            expt = dirichlet(qPi.prior.alpha, data_len).T
        return expt

    @classmethod
    def expt_by_kmeans(cls, wfs, n_states):
        data_len = wfs.shape[0]
        data_dim = wfs.shape[-1]
        if data_dim != 1:
            on_idx = wfs.max(0) >= cls.off_th
        else:
            on_idx = wfs[:, 0] >= cls.off_th
        if wfs[on_idx].shape[0] > n_states:
            src = wfs[on_idx]
        else:
            src = wfs
            on_idx = ones(data_len, dtype=bool)
        if src.shape[0] == data_len:
            kmeans = KMeans(n_clusters=n_states)
            lbl_shift = 0
        else:
            kmeans = KMeans(n_clusters=n_states - 1)
            lbl_shift = 1
        kmeans.fit(src)
        cat_array = zeros(data_len)
        cat_array[on_idx] = kmeans.labels_ + lbl_shift
        cat_array = cat_array.astype(int)
        # expt_s_eps = 1e-2
        expt_s_eps = cls.eps
        expt_s = ones((n_states, data_len)) * expt_s_eps
        for k in xrange(n_states):
            expt_s[k, cat_array == k] = 1
        expt_s = expt_s / expt_s.sum(0)[newaxis, :]
        return expt_s


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


if __name__ == '__main__':
    pass
