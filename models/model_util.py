#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
model_util.py
'''
import os
import sys

from numpy import round as nround

cdir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(cdir, '..'))
from util.logger import logger


class CheckTools(object):
    @classmethod
    def log_info_update_itr(cls, max_itr, itr, msg='', interval_digit=1):
        len_digit = len('%d' % max_itr)
        if len_digit - interval_digit < 0:
            interval_digit = 0
        interval = 10**interval_digit
        if itr % interval == 0 or itr == (max_itr - 1):
            logger.info(
                '%s update :%5d / %5d (interval %d)' %
                (msg, itr, max_itr, interval))
        logger.debug('%s update :%5d / %5d' % (msg, itr, max_itr))

    @classmethod
    def check_vb_increase(cls, vbs, i, decimals=0):
        if i < 1:
            return True
        vb_prv = nround(vbs[i - 1], decimals=decimals)
        vb_cur = nround(vbs[i], decimals=decimals)
        if vb_prv > vb_cur:
            logger.error(
                'vb decreased. iter:%d, %.10f(%.10f->%.10f)' %
                (i, vbs[i - 1] - vbs[i], vbs[i - 1], vbs[i]))
            return False
        else:
            return True


if __name__ == '__main__':
    pass
