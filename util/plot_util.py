#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
plot_util.py
'''

from itertools import cycle
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class StateColor:
    def __init__(self, n_states, **argvs):
        '''
        sc = StateColor(n_states)
        col = sc.get_color(k)
        col_list = sc.get_color_list()
        '''
        cmap_name = argvs.get('cmap_name', 'jet')
        # cmap_name = argvs.get('cmap_name', 'hot')
        self.alternate = argvs.get('alternate', True)
        self.n_states = n_states
        self.color_list = None
        self.col_map = plt.get_cmap(cmap_name)
        eps = 1e-10
        self.denom = float(n_states - 1) + eps
        self._prepare_color_list()

    def _prepare_color_list(self):
        dst = [self.col_map(k / (self.denom)) for k in range(self.n_states)]
        if self.alternate:
            tmp = [None] * len(dst)
            tmp[::2] = dst[::2]
            tmp[1::2] = dst[1::2][::-1]
            dst = tmp
        self.color_list = dst

    def get_color(self, k):
        return self.color_list[k]

    def get_color_list(self, alternate=True):
        if self.alternate != alternate:
            self._prepare_color_list()
        return self.color_list


class Markers():
    '''
    mks = Markers()
    mc_filled = mks.filled()
    mc_filled_list = mks.get_filled_maker_list(n_mks)
    mc_unfilled = mks.unfilled()
    mc_all = mks.all()
    '''

    def __init__(self, **argvs):
        '''
        @argvs
        ignore_markers: list of markers which are wanted to beignored
        '''
        self.ignore_markers = argvs.get('ignore_markers', [])

    def filled(self):
        '''
        '''
        return cycle(Line2D.filled_markers)

    def unfilled(self):
        '''
        '''
        dst = cycle(
            [
                m for m, f in Line2D.markers.iteritems()
                if f != 'nothing' and not isinstance(m, int) and m not in self.
                ignore_markers and m not in Line2D.filled_markers
            ])
        return dst

    def all(self):
        '''
        '''
        dst = cycle(
            [
                m for m, f in Line2D.markers.iteritems()
                if m not in self.ignore_markers
            ])
        return dst

    def get_filled_marker_list(self, n_mks):
        '''
        n_mks: number of makers
        '''
        m = self.filled()
        dst = [m.next() for nn in range(n_mks)]
        return dst


class ColorMarkers():
    '''
    cms = ColorMarkers(n_total)
    c, m = cms.get_color_marker(n)
    '''

    def __init__(self, n_total):
        '''
        @argvs
        n_total: number of makers and colors
        '''
        self.sc = StateColor(n_total)
        self.mk = Markers(ignore_markers=['*']).get_filled_marker_list(n_total)

    def get_color_marker(self, n):
        '''
        n: index out of n_total
        '''
        c = self.sc.get_color(n)
        m = self.mk[n]
        return c, m
