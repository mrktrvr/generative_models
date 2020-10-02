#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
plot_models.py
'''
import os
import sys

from numpy import atleast_2d
from numpy import array as arr
from numpy import arange
from numpy import unique
from numpy import zeros
from numpy import ones
from numpy import nonzero
from numpy import nanmax
from numpy import degrees
from numpy import arctan
from numpy import sqrt
from numpy.linalg import eigh
from numpy.linalg import norm
from numpy.random import randn
from numpy.random import choice

from IPython import embed

from matplotlib import pyplot as plt

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))
from util.plot_util import StateColor
from util.logger import logger
from util.time_util import TimeUtil


class PlotModelsGrid():
    def __init__(self, n_col, n_row, n_clusters):
        from numpy import ceil
        if n_clusters > n_col * n_row:
            self.n_col = int(ceil(n_clusters / float(n_col)))
        else:
            self.n_col = n_col
        self.n_row = n_row
        self.n_clusters = n_clusters
        self.shape = (n_row, n_col)
        self.figsize = (3 * self.n_col, 2 * self.n_row)

    def _idx_itr(self):
        for k in range(self.n_clusters):
            i = k % self.n_col
            j = int(k / self.n_row)
            yield i, j, k

    def _text_not_exist(self, ax):
        ax.text(0, 0, 'not exist', va='center', ha='center')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    def _plot_each_grid(self, ax, x):
        ax.plot(x)
        ax.grid(True)

    def __call__(self, w, s, figno=3):
        '''
        w: np.array(data_dim, n_clusters)
        s: np.array(data_len), array of cluster inditators
        figno: figno, default 3
        '''
        fig = plt.figure(figno, figsize=self.figsize)
        plt.clf()
        for i, j, k in self._idx_itr():
            ax = plt.subplot2grid(self.shape, (j, i), fig=fig)
            x = w[:, s == k]
            if x.shape[-1] == 0:
                self._text_not_exist(ax)
            else:
                self._plot_each_grid(ax, x)
            plt.title('state %d, %d' % (k, x.shape[-1]))
        plt.tightb_layout()


class PlotModels():
    '''
    '''

    def __init__(self, n_row, n_col, figno=10, **argvs):
        '''
        n_row: number of rows in suplot2grid
        n_col: number of columns in suplot2grid
        figno:
        '''
        self.n_row = n_row
        self.n_col = n_col
        self.figno = figno
        xmax = argvs.get('xmax', None)
        ymax = argvs.get('ymax', None)
        sizex = argvs.get('sizex', 5)
        sizey = argvs.get('sizey', 3)
        self.figsize = (self.n_col * sizex, self.n_row * sizey)
        self.grid_shape = (self.n_row, self.n_col)
        self.xmax = xmax
        self.ymax = ymax
        self.fig = None
        self.f_size = 'xx-small'
        self.ncol_leg = 8
        self.clf()

    def _decos_grid(self, **argvs):
        self.ax.grid(True)
        xlim = argvs.get('xlim', None)
        ylim = argvs.get('ylim', None)
        if xlim is not None:
            if xlim[0] == xlim[1]:
                v = 10
                logstr = 'xlim modified %s ->' % list(xlim)
                xlim = (xlim[0] - v, xlim[1] + v)
                logger.info('%s %s' % (logstr, list(xlim)))
            self.ax.set_xlim(xlim)
        if ylim is not None:
            if ylim[0] == ylim[1]:
                logstr = 'ylim modified %s ->' % list(ylim)
                v = 10
                ylim = (ylim[0] - v, ylim[1] + v)
                logger.info('%s %s' % (logstr, list(ylim)))
            self.ax.set_ylim(ylim)
        self.ax.tick_params(axis='both', labelsize=6)

    def _decos_str(self, **argvs):
        title = argvs.get('title', '')
        xlbl = argvs.get('xlbl', '')
        ylbl = argvs.get('ylbl', '')
        do_leg = argvs.get('do_leg', False)
        lbls = argvs.get('lbls', None)
        if title != '':
            self.ax.set_title(title, fontsize=self.f_size)
        if xlbl != '':
            self.ax.set_xlabel(xlbl, fontsize=self.f_size)
        if ylbl != '':
            self.ax.set_ylabel(ylbl, fontsize=self.f_size)
        if do_leg:
            if lbls is None:
                self.ax.legend(loc=0, fontsize=self.f_size, ncol=self.ncol_leg)
            else:
                self.ax.legend(
                    lbls, loc=0, fontsize=self.f_size, ncol=self.ncol_leg)

    def _decos_yticks(self, n_states, interval=3):
        ticks = arange(0, n_states, interval) + 0.5
        self.ax.set_yticks(ticks)
        ticklbls = arange(0, n_states, interval)
        self.ax.set_yticklabels(ticklbls, va='center')
        self.ax.set_ylim(0, n_states)

    def _decos_xticks(self, pltx):
        xtick_step = 3600
        self.ax.set_xlim(pltx[0], pltx[-1])
        xtck = pltx[::xtick_step].tolist()
        xtck += [pltx[-1] + 1]
        if pltx[0] > 1483228800:
            tu = TimeUtil()
            xtck_lbls = [tu.ut2ts(x) for x in pltx[::xtick_step]]
            xtck_lbls += [tu.ut2ts(pltx[-1] + 1)]
        else:
            xtck_lbls = [x for x in pltx[::xtick_step]]
            xtck_lbls += [pltx[-1] + 1]
        self.ax.set_xticks(xtck)
        self.ax.set_xticklabels(xtck_lbls, rotation=90, fontsize=self.f_size)

    def clf(self):
        if self.fig is not None:
            plt.clf()
        self.fig = plt.figure(self.figno, figsize=self.figsize)

    def ion_show(self):
        plt.ion()
        plt.show()

    def show(self):
        plt.show()

    def sup_title(self, sup_title):
        plt.suptitle(sup_title)

    def tight_layout(self):
        try:
            plt.tight_layout()
        except Exception as e:
            logger.error('tight_layout %s' % e)

    def savefig(self, fig_name, overwrite=True):
        if os.path.exists(fig_name) and overwrite is False:
            logger.info('%s exists. savefig skipped' % fig_name)
        else:
            plt.savefig(fig_name)
            logger.info('%s saved' % fig_name)

    def close_all(self):
        plt.close('all')

    def get_ax(self, pos, rspan=1, cspan=1):
        ax = plt.subplot2grid(
            self.grid_shape, pos, rowspan=rspan, colspan=cspan, fig=self.fig)
        return ax

    def plot_2d_scatter(self, pos, src, mu, cat=None, cov=None, **args):
        '''
        pos: position of subplot2grid. (int, int)
        mu: np.array(data_dim, n_states)
        cov: np.array(data_dim, data_dim, n_states)
        '''
        logger.info('started')
        cspan = args.get('cspan', 1)
        rspan = args.get('rspan', 1)
        xlbl = args.get('xlbl', '')
        ylbl = args.get('ylbl', '')
        title = args.get('title', '')
        idx1 = args.get('idx1', 0)
        idx2 = args.get('idx2', 1)
        lbls = args.get('lbls', None)
        if src.shape[0] == 1:
            idx1 = idx2 = 0
        xlim = (src[idx1].min(), src[idx1].max())
        ylim = (src[idx2].min(), src[idx2].max())
        self.ax = self.get_ax(pos, rspan=rspan, cspan=cspan)
        self._plot_scatter_core(src, mu, idx1, idx2, cat)
        self._decos_str(title=title, xlbl=xlbl, ylbl=ylbl)
        self._decos_grid(xlim=xlim, ylim=ylim, lbls=lbls)
        logger.info('done')

    def _plot_scatter_core(self, src, mu, idx1, idx2, cat, mk=None, ms=0.5):
        mk = '.' if mk is None else mk
        n_states = mu.shape[-1] if mu.ndim == 2 else 1
        clr = StateColor(n_states).get_color_list()
        for k in range(n_states):
            x, y = mu[idx1, k], mu[idx2, k]
            self.ax.plot(x, y, 'x', ms=10, color=clr[k], label='state %d' % k)
        self.ax.legend(loc=0)
        if cat is None:
            self.ax.plot(src[idx1, :], src[idx2, :], mk, alpha=0.1, ms=ms)
        else:
            s = cat.argmax(0) if cat.ndim == 2 else cat
            for k in range(n_states):
                x, y = src[idx1, s == k], src[idx2, s == k]
                self.ax.plot(x, y, mk, color=clr[k], alpha=0.5, ms=ms)

    def _calc_ellipse_info(self, mu, cov, idx1, idx2):
        idx = arr([idx1, idx2])
        # ---
        eig_v, eig_w = eigh(cov)
        # --- angle
        norm_eig_w = eig_w[0] / norm(eig_w[0])
        rad = arctan(norm_eig_w[idx2] / norm_eig_w[idx1])
        deg = degrees(rad) + 180.0
        # --- width and height
        v = 2.0 * sqrt(2.0) * sqrt(eig_v)
        w, h = v[idx1], v[idx2]
        h *= 1e+1
        w *= 1e+1
        m = mu[idx]
        return m, w, h, deg

    def plot_2d_mu_cov(self, pos, mu, cov, src=None, cat=None, **args):
        '''
        pos: position of subplot2grid. (int, int)
        mu: np.array(data_dim, n_states)
        cov: np.array(data_dim, data_dim, n_states)
        '''
        from matplotlib.patches import Ellipse
        cspan = args.get('cspan', 1)
        rspan = args.get('rspan', 1)
        xlbl = args.get('xlbl', '')
        ylbl = args.get('ylbl', '')
        idx1 = args.get('idx1', 0)
        idx2 = args.get('idx2', 1)
        title = args.get('title', 'Dim %d v %d' % (idx1, idx2))
        lbls = args.get('lbls', None)
        n_states = mu.shape[-1] if mu.ndim == 2 else 1
        clr = StateColor(n_states).get_color_list()
        if src is None:
            xlim = (mu.min() - sqrt(cov.max()), mu.max() + sqrt(cov.max()))
            ylim = xlim
        else:
            xlim = (src[idx1].min(), src[idx1].max())
            ylim = (src[idx2].min(), src[idx2].max())
        self.ax = self.get_ax(pos, rspan=rspan, cspan=cspan)
        # --- src scatter
        if src is not None:
            self._plot_scatter_core(src, mu, idx1, idx2, cat)
        else:
            for k in range(n_states):
                x, y = mu[idx1, k], mu[idx2, k]
                self.ax.plot(
                    x, y, 'x', ms=10, color=clr[k], label='state %d' % k)
                self.ax.legend(loc=0)

        # --- mu and cov
        for k in range(n_states):
            ret = self._calc_ellipse_info(mu[:, k], cov[:, :, k], idx1, idx2)
            m, w, h, d = ret
            ellipse = Ellipse(m, w, h, d, color=clr[k], alpha=0.2)
            self.ax.add_artist(ellipse)
        self._decos_str(title=title, xlbl=xlbl, ylbl=ylbl)
        self._decos_grid(xlim=xlim, ylim=ylim, lbls=lbls)

    def plot_2d_array(self, pos, src, **args):
        '''
        pos: position of subplot2grid. (int, int)
        src: np.array(x_size, y_size)
        '''
        xlbl = args.get('xlbl', '')
        ylbl = args.get('ylbl', '')
        title = args.get('title', '')
        cspan = args.get('cspan', 1)
        rspan = args.get('rspan', 1)
        baseline = args.get('baseline', 'sym')
        lbls = args.get('lbls', ['%d' % x for x in range(src.shape[-1])])
        self.ax = self.get_ax(pos, rspan=rspan, cspan=cspan)
        n_states = src.shape[-1]
        if n_states < 100:
            clr = StateColor(n_states).get_color_list()
            for k in range(n_states):
                self.ax.plot(src[:, k], color=clr[k])
        else:
            self.ax.plot(src)
        # self.ax.legend(lbls, loc=0)
        xlim = (-1, src.shape[0])
        ymax = nanmax(abs(src))
        if baseline == 'zero':
            ylim = (0, ymax)
        else:
            ylim = (-ymax, ymax)
        self._decos_str(title=title, xlbl=xlbl, ylbl=ylbl)
        self._decos_grid(xlim=xlim, ylim=ylim, lbls=lbls)

    def plot_seq(self, pos, src, cat=None, pltx=None, **args):
        '''
        pos: position of subplot2grid. (int, int)
        src: np.array(data_dim, data_len) or np.array(data_len)
        cat: np.array(n_states, data_len)
        '''
        logger.info('started')
        title = args.get('title', '')
        lbls = args.get('lbls', None)
        cspan = args.get('cspan', self.n_col)
        rspan = args.get('rspan', 1)
        ymin = args.get('ymin', None)
        xlbl = args.get('xlbl', '')
        ylbl = args.get('ylbl', '')
        ymin = src.min() if ymin is None else ymin
        ymax = nanmax(abs(src))
        src = atleast_2d(src)
        n_src, data_len = src.shape
        self.ax = self.get_ax(pos, rspan=rspan, cspan=cspan)
        pltx = arange(data_len) if pltx is None else pltx
        for v in src:
            self.ax.plot(pltx, v, label='sequence')
        if cat is not None:
            if cat.ndim == 1:
                n_states = int(nanmax(unique(cat))) + 1
                y = zeros((n_states, data_len))
                for k in unique(cat):
                    y[k, cat == k] = 1
            else:
                n_states = cat.shape[0]
                y = cat
            lbls = ['%d' % k
                    for k in range(n_states)] if lbls is None else lbls
            clr = StateColor(n_states).get_color_list()
            y *= ymax
            self.ax.stackplot(
                pltx, y, colors=clr, alpha=0.5, labels=lbls, baseline='zero')
        xlim = (pltx[0], pltx[-1])
        ylim = (ymin, ymax)
        self._decos_xticks(pltx)
        self._decos_str(title=title, xlbl=xlbl, ylbl=ylbl)
        self._decos_grid(xlim=xlim, ylim=ylim)
        logger.info('done')

    def _reshape_cat(self, cat):
        if cat.ndim == 2:
            n_states, data_len = cat.shape
            expt = cat
            sidx = expt.argmax(0)
        else:
            data_len = len(cat)
            n_states = nanmax(unique(cat)) + 1
            expt = zeros((n_states, data_len))
            for k in unique(cat):
                expt[k, cat == k] = 1
            sidx = cat
        return expt, sidx, data_len, n_states

    def plot_states_stack(self, pos, cat, pltx=None, ymax=1, **args):
        title = args.get('title', '')
        cspan = args.get('cspan', self.n_col)
        rspan = args.get('rspan', 1)
        self.ax = self.get_ax(pos, rspan=rspan, cspan=cspan)
        if cat.ndim == 1:
            expt, sidx, data_len, n_states = self._reshape_cat(cat)
        else:
            expt = cat
            sidx = cat.argmax(0)
            n_states, data_len = cat.shape
        pltx = arange(data_len) if pltx is None else pltx
        v = expt * ymax
        colors = StateColor(n_states).get_color_list()
        lbls = ['%d(%d)' % (k, len(sidx[sidx == k])) for k in range(n_states)]
        self.ax.stackplot(
            pltx, v, colors=colors, alpha=0.5, labels=lbls, baseline='zero')
        xlbl = 'sequence'
        ylbl = 'amplitude'
        xlim = (pltx[0], pltx[-1])
        ylim = (0, ymax)
        self._decos_str(title=title, xlbl=xlbl, ylbl=ylbl)
        self._decos_grid(xlim=xlim, ylim=ylim)

    def plot_states_indv(self, pos, cat, pltx=None, pws=None, **args):
        '''
        pm.plot_states_indv(ps, cat, pltx=None, pws=None)
        @argvs
        pos: position of subplot2axis. (row, column)
        cat: np.array(n_states, data_len)
        pltx: np.array(data_len)
        pws: np.array(data_dim, data_len)
        '''
        title = args.get('title', '')
        cspan = args.get('cspan', self.n_col)
        rspan = args.get('rspan', 1)
        cspan = self.n_col if cspan == -1 else cspan
        self.ax = self.get_ax(pos, rspan=rspan, cspan=cspan)
        expt, sidx, data_len, n_states = self._reshape_cat(cat)
        sc_cls = StateColor(n_states)
        cols = sc_cls.get_color_list(True)
        pltx = arange(data_len) if pltx is None else pltx
        btm = ones(data_len)
        for k in range(n_states):
            c = cols[k]
            top = ones(data_len) * k
            # top[sidx == k] += 1
            top += expt[k]
            x, y, l = pltx, btm * k, '%d (%6d)' % (k, sidx[sidx == k].shape[0])
            self.ax.fill_between(x, y, top, label=l, color=c, alpha=0.5, lw=0)
            if top[top == k + 1].shape[0] == 0:
                continue
            x, y, v = pltx[nonzero(top == k + 1)[0][0]], k + 0.5, '%2d->' % k
            ha, va = 'right', 'center'
            self.ax.text(x, y, v, ha=ha, va=va, fontsize=self.f_size, color=c)
            x, y, v = pltx[nonzero(top == k + 1)[0][-1]], k + 0.5, '<-%2d' % k
            ha, va = 'left', 'center'
            self.ax.text(x, y, v, ha=ha, va=va, fontsize=self.f_size, color=c)
        if pws is not None:
            plt_pws = n_states * pws / nanmax(pws)
            self.ax.plot(pltx, plt_pws.T, color='black', lw=0.3)
        # --- decos
        yticklbls = ['%2d' % k for k in range(0, n_states)]
        self.ax.set_yticks(range(0, n_states))
        self.ax.set_yticklabels(yticklbls, fontsize=self.f_size, va='bottom')
        title_str = '%s states (n_states: %d)' % (title, n_states)
        self._decos_str(title=title_str)
        self._decos_xticks(pltx)
        self._decos_yticks(n_states)
        xlim = (pltx[0], pltx[-1])
        self._decos_grid(xlim=xlim)

    def multi_bar(self, pos, src, **args):
        logger.info('started')
        xticks = args.get('xticks', None)
        ylbl = args.get('ylbl', '')
        xlbl = args.get('xlbl', '')
        lbls = args.get('lbls', None)
        title = args.get('title', '')
        cspan = args.get('cspan', 1)
        rspan = args.get('rspan', 1)
        leg_list = args.get('leg_list', None)
        margin = 0.05
        self.ax = self.get_ax(pos, rspan=rspan, cspan=cspan)
        n_bars, n_multi = src.shape
        w = (1.0 - 2 * margin) / n_multi
        clist = StateColor(n_multi).get_color_list()
        for i in range(n_multi):
            shift = (w / 2.0 + w * i)
            x = arange(n_bars) + shift + margin
            y = src[:, i]
            lbl = '%d' % i
            self.ax.bar(x, y, width=w, label=lbl, color=clist[i], alpha=0.5)
        if leg_list is not None:
            self.ax.legend(leg_list, fontsize=self.f_size)
        # --- splitter
        ymax = nanmax(src)
        for j in range(n_bars):
            self.ax.plot([j, j], [0, ymax], 'k--', lw=0.3)
        xticks = arange(n_bars) if xticks is None else xticks
        self.ax.set_xticks(arange(n_bars) + 0.5)
        self.ax.set_xticklabels(xticks, fontsize=self.f_size)
        ylim = (0, ymax)
        xlim = (0, n_bars)
        self._decos_str(title=title, xlbl=xlbl, ylbl=ylbl, lbls=lbls)
        self._decos_grid(xlim=xlim, ylim=ylim)
        self.ax.grid(False)
        self.ax.yaxis.grid(True)
        logger.info('done')

    def plot_table(self, pos, src, **args):
        '''
        src: n_states x n_states
        fmt: '%8.2f'
        '''
        fmt = args.get('fmt', '%8.2f')
        cspan = args.get('cspan', 1)
        rspan = args.get('rspan', 1)
        self.ax = self.get_ax(pos, rspan=rspan, cspan=cspan)
        ndim = src.ndim
        n_states = src.shape[0]
        rows = ['k:%d' % x for x in range(n_states)]
        if ndim == 1:
            cell = [[fmt % a for a in src]]
            cols = None
            colw = ones(len(cell[0])) * 0.6
        elif ndim == 2:
            cell = [[fmt % v for v in src[k]] for k in range(n_states)]
            if True:
                cell.append([fmt % v for v in src.sum(0)])
                rows += ['sum']
            cols = ['prv k:%d' % xx for xx in range(n_states)]
            colw = ones(n_states) / float(n_states)
        else:
            logger.error('ndim %d not supported' % ndim)
        self.ax.table(
            cellText=cell,
            rowLabels=rows,
            colLabels=cols,
            colWidths=colw,
            loc='center',
            cellLoc='center',
            colLoc='center')
        self.ax.set_axis_off()

    def plot_vb(self, pos, vbs, **args):
        from numpy import isnan
        from numpy import any as nany
        from numpy import diff
        cspan = args.get('cspan', 1)
        rspan = args.get('rspan', 1)
        len_vb = len(vbs)
        self.ax = self.get_ax(pos, rspan=rspan, cspan=cspan)
        if not nany(isnan(vbs)):
            self.ax.plot(vbs, label='vbs')
            vb_df = diff(vbs)
            for i, v in enumerate(vb_df, 1):
                if v < 0:
                    self.ax.plot(i, v, 'rx', label='%d' % i)
            xlim = (0, len_vb)
            ylim = (vbs.min(), vbs.max() + 10)
        else:
            self.ax.text(0, 0, 'vbs contains nan', va='center', ha='center')
            xlim = (-10, 10)
            ylim = (-10, 10)
        title = 'variational bound (%d)' % len_vb
        self._decos_str(title=title)
        self._decos_grid(xlim=xlim, ylim=ylim)

    def __call__(self, Y, S, Z, mu, mu_h):
        self.plot2d((0, 0), mu, arange(mu.shape[-1]), title=r'$\mu$-S')
        self.plot2d((0, 1), mu_h, arange(mu.shape[-1]), title=r'$\hat{\mu}$-S')
        self.plot2d((0, 2), Y, S, title='data - states')
        self.plot2d((0, 3), Y, Z, title='data - states')
        # self.plot_seq((1, 0), Y, S, title='data - states')
        self.plot_states_seq((1, 0), S, title='data - states')
        self.plot_seq((2, 0), Y, Z, title='data - category')
        plt.tight_layout()


def plot_hmm(Y, expt_s, mu, pi, A, figno):
    pm = PlotModels(2, 3, figno)
    pm.plot_seq(Y.max(1), expt_s)
    pass


def main_plot_models():
    data_len = 100
    data_dim = 2
    n_states = 4
    mu = randn(data_dim, n_states) * 10
    Y = randn(data_dim, data_len) * 10
    S = choice(arange(n_states), size=data_len)
    n_row, n_col = 3, 4
    PlotModels(n_row, n_col)(Y, S, S, mu, mu)
    embed(header='main_plot_models')


def main_plot_multi_bar():
    from numpy import ones
    from numpy.random import dirichlet
    n_states = 8
    n_cat = 3
    pi = dirichlet(ones(n_states), size=n_cat).T
    print(pi)
    pm = PlotModels(1, 1, 1)
    pm.multi_bar((0, 0), pi)
    embed(header='main_plot_multi_bar')


def main_plot_models_grid():
    data_len = 100
    data_dim = 10
    n_states = 16
    Y = randn(data_dim, data_len) * 10
    S = choice(arange(n_states), size=data_len)
    n_row, n_col = 4, 4
    PlotModelsGrid(n_row, n_col, n_states)(Y, S)
    embed()


if __name__ == '__main__':
    main_plot_models()
    # main_plot_models_grid()
    # main_plot_multi_bar()
