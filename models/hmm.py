#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
hmm.py
'''
import os
import sys
import cPickle as pickle

from numpy import nan
from numpy import atleast_2d
from numpy import ones
from numpy import zeros
from numpy import exp
from numpy import einsum
from numpy.random import choice
from numpy.random import multivariate_normal as mvnrand
from numpy.random import dirichlet

from gmm import Theta as GmmTheta
from gmm import qPi as GmmPi
from gmm import qS as GmmS
from forward_backward import fb_alg
from model_util import CheckTools

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))
from distributions.dirichlet import Dirichlet
from distribution.kl_divergence import KL_Dir
from util.calc_util import inv
from util.logger import logger


class qPi(GmmPi):
    '''
    pi = qPi(n_states)
    pi.prior.expt: arr(n_states)
    pi.post.expt: arr(n_states)
    pi.update(s)
    alpha: array(n_states)
    '''

    def __init__(self, n_states):
        '''
        pi = qPi(n_states)
        '''
        super(qPi, self).__init__(n_states)

    def update(self, S):
        '''
        qpi.update(S)
        @argvs
        S: <S>, np.array(n_states, data_len)
        '''
        alpha_h = self.prior.alpha + S[:, 0]
        self.post.set_params(alpha=alpha_h)


class qA(GmmPi):
    '''
    A = qA(n_states)
    '''

    def __init__(self, n_states):
        '''
        A = qA(n_states)
        '''
        super(qA, self).__init__(n_states)

    def set_default_params(self):
        '''
        A.set_default_parmas()
        '''
        self.set_params()

    def set_params(self, **argvs):
        '''
        A.set_params()
        '''
        self.prior = Dirichlet(self.n_states, self.n_states)
        self.prior.set_params(**argvs)
        self.post = Dirichlet(self.n_states, self.n_states)
        self.post.set_params(**argvs)

    def update(self, SS):
        '''
        qA.update(SS)
        @argvs
        SS: np.array(n_states, n_states, data_len)
        '''
        alpha_h = self.prior.alpha + SS[:, :, 1:].sum(2)
        self.post.set_params(alpha=alpha_h)

    def calc_kl_divergence(self):
        kl_dir = 0
        for k in xrange(self.n_states):
            kl_dir += KL_Dir(self.post.ln_alpha[k], self.prior.ln_alpha[k])
        return kl_dir


class Theta(GmmTheta):
    def __init__(self, data_dim, n_states):
        super(Theta, self).__init__(data_dim, n_states)
        self.qpi = qPi(self.n_states)
        self.qA = qA(self.n_states)
        self.update_order = ['pi', 'A', 'MuR']

    def set_default_params(self):
        self.qmur.set_default_params()
        self.qpi.set_default_params()
        self.qA.set_default_params()

    def clear_params(self):
        self.qmur.clear_params()
        self.qpi.clear_params()

    def set_params(self, prm):
        '''
        @args
        prm: dictionary
        {'MuR': {
            'mu':: (data_dim, n_states),
            'beta': (n_states)
            'nu': (n_states),
            'W': (data_dim, data_dim, n_states)
            },
         'Pi': {
            'alpha': (n_states),
            'ln_alpha': (n_states)
            },
         'A': {
            'alpha': (n_states, n_states),
            'ln_alpha': (n_states, n_states)
            },
        '''
        if 'MuR' in prm:
            self.qmur.set_params(**prm['MuR'])
        else:
            if self.qmur.prior is None:
                self.qmur.set_default_params()
        if 'Pi' in prm:
            self.qpi.set_params(**prm['Pi'])
        else:
            if self.qpi.prior is None:
                self.qpi.set_default_params()
        if 'A' in prm:
            self.qA.set_params(**prm['A'])
        else:
            if self.qA.prior is None:
                self.qA.set_default_params()
        self.n_states = len(self.qpi.post.alpha)

    def update(self, Y, s, ss, YY=None):
        '''
        theta.update(Y, s, ss, YY=None)
        '''
        for ut in self.update_order:
            if ut == 'pi':
                self.qpi.update(s)
            elif ut == 'A':
                self.qA.update(ss)
            elif ut == 'MuR':
                self.qmur.update(Y, s, YY=YY)
            else:
                logger.error('%s is not supported' % ut)

    def get_samples(self, data_len=1, by_posterior=True):
        '''
        theta.get_samples(data_len=1, by_posterior=True)
        @return
        mu:
        R:
        pi:
        A:
        '''
        mu, R = self.qmur.get_samples(data_len, by_posterior)
        pi = self.qpi.get_samples(data_len, by_posterior)
        A = self.qA.get_samples(data_len, by_posterior)
        return mu, R, pi, A

    def get_param_dict(self, by_posterior=True):
        '''
        theta.get_param_dict(by_posterior=True)
        @argvs
        by_posterior: use posterior params(True) or not(False)
        @return
        dst = {
        'MuR': {
            'mu': (data_dim, n_states),
            'beta': (n_states),
            'nu': (n_states),
            'W': (data_dim, data_dim, n_states),
            },
        'Pi': {
            'alpha': (n_states),
            'ln_alpha': (n_states),
            },
        'A': {
            'alpha': (n_states, n_states),
            'ln_alpha': (n_states, n_states),
            },
        'n_states': int,
        'data_dim': int,
        }
        '''
        dst = {
            'MuR': self.qmur.get_param_dict(by_posterior),
            'Pi': self.qpi.get_param_dict(by_posterior),
            'A': self.qA.get_param_dict(by_posterior),
            'n_states': self.n_states,
            'data_dim': self.data_dim,
        }
        return dst

    def save_param_dict(self, file_name, by_posterior=True):
        prm = self.get_param_dict(by_posterior)
        try:
            with open(file_name, 'w') as f:
                pickle.dump(prm, f)
            return True
        except:
            return False

    def load_param_dict(self, file_name):
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                prm = pickle.load(f)
            self.set_params(prm)
            return True
        else:
            return False


class qS(GmmS):
    '''
    s = qS(n_states)
    expt: (n_states, data_lem)
    expt2: (n_states, n_states, data_lem)
    '''

    def __init__(self, n_states, **argvs):
        super(qS, self).__init__(n_states)
        self.expt2 = None

    def init_expt(self, data_len):
        '''
        qs.init_expt(data_len)
        @argvs
        data_len: int
        @self
        expt: (n_states, data_lem)
        '''
        alpha_pi = ones(self.n_states)
        alpha_A = ones((self.n_states, self.n_states))
        expt = ones((self.n_states, data_len)) * nan
        expt[:, 0] = dirichlet(alpha_pi)
        for t in xrange(1, data_len):
            k_prev = choice(self.n_states, p=expt[:, t - 1])
            expt[:, t] = dirichlet(alpha_A[:, k_prev])
        self.set_expt(expt)

    def set_expt(self, expt, expt2=None):
        '''
        qs.set_expt(expt, expt2=None)
        @argvs
        expt: (n_states, data_lem)
        expt2: (n_states, n_states, data_lem)
        '''
        self.expt = expt
        if expt2 is None:
            expt2 = einsum('kt,jt->kjt', self.expt[:, :-1], self.expt[:, 1:])
        self.expt2 = expt2

    def clear_expt(self):
        '''
        qs.clear_expt()
        '''
        self.expt = None
        self.expt2 = None

    def update(self, Y, theta, YY=None):
        '''
        qs.update(Y, theta, YY=None)
        @argvs
        Y: observation data, np.array(data_dim, data_len)
        theta: class object
        YY: YY^T, np.array(data_dim, data_dim, data_len)
        '''
        logger.debug('calc ln_lkh')
        ln_lkh_gmm = theta.qmur.calc_ln_lkh(Y, YY)
        logger.debug('forward backward')
        lns, lnss, c = fb_alg(
            ln_lkh_gmm.T, theta.qpi.post.expt_ln, theta.qA.post.expt_ln)
        logger.debug('expt')
        s = exp(lns)
        ss = exp(lnss)
        self.expt = s
        self.expt2 = ss
        self.const = c

    def get_samples(self, data_len, pi, A):
        '''
        qS.get_samples(data_len, pi, A, use_uniform_s=False)
        @argvs
        data_len: int
        pi: np.array(n_states)
        A: np.array(n_states, n_states)
        '''
        S = zeros(data_len, dtype=int)
        for t in xrange(data_len):
            if t == 0:
                k = choice(self.n_states, p=pi)
            else:
                k = choice(self.n_states, p=A[:, S[t - 1]])
            S[t] = k
        return S


class Hmm(CheckTools):
    '''
    hmm = Hmm(data_dim, n_states)
    hmm.set_params(prms)
    hmm.get_samples(data_len)
    hmm.init_expt_s(data_le)
    hmm.update(Y)
    '''

    def __init__(self, data_dim, n_states, **argvs):
        '''
        '''
        self.data_dim = data_dim
        self.n_states = n_states
        self.max_em_itr = argvs.get('max_em_itr', 20)
        self.theta = Theta(data_dim, n_states)
        self.qs = qS(n_states)
        self.expt_s = None
        self.update_order = ['M', 'E']
        self.vbs = zeros(self.max_em_itr)
        self.data_info = None

    def init_expt_s(self, data_len):
        '''
        hmm.init_expt_s(data_len)
        '''
        if self.qs.expt is None:
            self.qs.init_expt(data_len)
            self.expt_s = self.qs.expt

    def set_expt_s(self, expt, expt2=None):
        '''
        hmm.set_expt_s(expt, expt2=None)
        '''
        self.qs.set_expt(expt, expt2)
        self.expt_s = self.qs.expt

    def set_default_params(self):
        '''
        hmm.set_default_params()
        '''
        self.theta.set_default_params()

    def set_params(self, prm):
        '''
        {'MuR': {
            'mu':: (data_dim, n_states),
            'beta': (n_states)
            'nu': (n_states),
            'W': (data_dim, data_dim, n_states)
            },
         'Pi': {
            'alpha': (n_states),
            'ln_alpha': (n_states)
            },
        'A': {
            alpha: (n_states x n_states),
            ln_alpha: (n_states x n_states),
            }
        '''
        self.theta.set_params(prm)
        self.data_dim = self.theta.data_dim

    def clear(self):
        '''
        hmm.clear()
        '''
        self.theta.clear_params()
        self.qs.clear_expt()

    def update(self, Y):
        '''
        hmm.update(Y)
        '''
        self.data_dim, data_len = Y.shape
        self.init_expt_s(data_len)
        for i in xrange(self.max_em_itr):
            self.log_info_update_itr(self.max_em_itr, i, interval_digit=1)
            for j, uo in enumerate(self.update_order):
                if uo == 'E':
                    self.qs.update(Y, self.theta)
                elif uo == 'M':
                    self.theta.update(Y, self.qs.expt, self.qs.expt2)
                else:
                    logger.error('%s is not supported' % uo)
            self._update_vb(i)
        self.expt_s = self.qs.expt

    def _update_vb(self, i):
        self.vbs[i] = self.calc_vb()
        if not self.check_vb_increase(self.vbs, i):
            logger.warn('vb increased')
            # embed(header='update_vb')
            pass

    def calc_vb(self):
        kl_mur = self.theta.qmur.calc_kl_divergence()
        kl_pi = self.theta.qpi.calc_kl_divergence()
        kl_A = self.theta.qA.calc_kl_divergence()
        vb = self.qs.const - kl_pi - kl_A - kl_mur
        return vb

    def get_samples(self, data_len, by_posterior=True):
        '''
        Y, S, mu, R, pi, A = hmm.sample(data_len)
        @argvs
        data_len: data length
        use_uniform_s: use uniform S if True, use sampled S if False
        @return
        Y: sampled observations, np.array(data_dim, data_len)
        S: sampled states, np.array(n_states, data_len)
        Z: sampled categories, np.array(n_cat, data_len)
        prms: [mu, R, pi, A, psi]
            mu: sampled mu: np.array(data_dim, n_states)
            R: sampled R: np.array(data_dim, data_dim, n_states)
            pi: sampled pi: np.array(n_states)
            A: sampled A: np.array(n_states, n_states)
        '''
        mu, R, pi, A = self.theta.get_samples(1, by_posterior)
        S = self.qs.get_samples(data_len, pi, A)
        Y = zeros((self.data_dim, data_len))
        for t in xrange(data_len):
            k = S[t]
            cov = inv(R[:, :, k])
            Y[:, t] = mvnrand(mu[:, k], cov)
        return Y, S, [mu, R, pi, A]

    def get_estims(self, Y=None, use_sample=False):
        '''
        return
        estim_y: (data_dim, data_len), waveform sequence
        estim_s: (data_len), state sequence which contains 0 to n_states - 1
        vb: float value, valiational bound
        '''
        estim_s = self.qs.get_estims(Y, self.theta, True)
        estim_y = zeros((self.data_dim, len(estim_s)))
        if use_sample:
            for k in xrange(self.n_states):
                idx = estim_s == k
                data_len = estim_y[:, idx].shape[-1]
                mu, R = self.theta.qmur.post.sample()
                estim_y[:, idx] = mvnrand(
                    mu[:, k], inv(R[:, :, k]), size=data_len).T
        else:
            for k in xrange(self.n_states):
                idx = estim_s == k
                data_len = estim_y[:, idx].shape[-1]
                m = self.theta.qmur.post.mu[:, k]
                c = inv(self.theta.qmur.post.expt_prec[:, :, k])
                estim_y[:, idx] = mvnrand(m, c, size=data_len).T
        vb = self.calc_vb()
        return estim_y, estim_s, vb

    def get_param_dict(self, by_posterior=True):
        '''
        dst = {
        'MuR': {
            'mu': (data_dim, n_states),
            'beta': (n_states),
            'nu': (n_states),
            'W': (data_dim, data_dim, n_states),
            },
        'Pi': {
            'alpha': (n_states),
            'ln_alpha': (n_states),
            },
        'A': {
            'alpha': (n_states, n_states),
            'ln_alpha': (n_states, n_states),
            },
        'n_states': int,
        'data_dim': int,
        }
        '''
        dst = self.theta.get_param_dict(by_posterior)
        # self.data_dim = dst['data_dim']
        return dst

    def save_params(self, file_name):
        was_saved = self.theta.save_param_dict(file_name)
        logger.info('saved %s' % file_name)
        return was_saved

    def load_params(self, file_name):
        was_loaded = self.theta.load_param_dict(file_name)
        self.data_dim = self.theta.data_dim
        self.n_states = self.theta.n_states
        return was_loaded

    def save_expt_s(self, file_name):
        self.qs.save_estims(file_name)

    def load_expt_s(self, file_name):
        dst = self.qs.load_estims(file_name)
        self.expt_s = self.qs.expt
        return dst


def plotter(y, s, hmm, figno=1):
    daat_dim, data_len = y.shape
    # --- sample
    if False:
        _, _, prm = hmm.get_samples(data_len)
        vbs = hmm.vbs
        _plotter_core(y, s, prm, vbs, 'sample', 100 * figno + 1)
    # --- expectation
    if True:
        prm = [
            hmm.theta.qmur.post.mu,
            hmm.theta.qmur.post.expt_prec,
            hmm.theta.qpi.post.expt,
            hmm.theta.qA.post.expt,
        ]
        vbs = hmm.vbs
        _plotter_core(y, s, prm, vbs, 'expextation', 100 * figno + 2)


def _plotter_core(y, s, prm, vbs, prm_type_str, figno):
    from plot_models import PlotModels
    mu, r, pi, A = prm
    n_cols = 3
    pm = PlotModels(3, n_cols, figno)
    idx1, idx2 = 0, 1
    # --- params
    pm.plot_2d_array((0, 0), mu, title=r'$\mu$ %s' % prm_type_str)
    pm.multi_bar((0, 1), atleast_2d(pi), title=r'$\pi$ %s' % prm_type_str)
    pm.plot_table((0, 2), A)
    # cov = inv(r.transpose(2, 0, 1)).transpose(1, 2, 0)
    # pm.plot_2d_mu_cov((0, 2), mu, cov, src=y, cat=s)
    # --- Y
    title = 'Scatter Dim %d v %d' % (idx1, idx2)
    pm.plot_2d_scatter((1, 0), y, mu, cat=s, idx1=idx1, idx2=idx2, title=title)
    pm.plot_2d_array((1, 1), y, title='Y')
    pm.plot_vb((1, 2), vbs, cspan=1)
    pm.plot_seq((2, 0), y, cat=s, title='Y', cspan=n_cols)
    pm.tight_layout()


def gen_data(data_dim, n_states, data_len):
    from numpy.random import randn
    from scipy.stats import wishart
    hmm = Hmm(data_dim, n_states, expt_init_mode='random')
    hmm.set_default_params()
    mu = randn(data_dim, n_states) * 10
    c = 1
    if False:
        from numpy import eye
        W = wishart.rvs(data_dim, c * eye(data_dim), size=n_states)
    else:
        W = wishart.rvs(data_dim, c * ones(data_dim), size=n_states)
    W = W.transpose(1, 2, 0)
    hmm.set_params({'MuR': {'mu': mu, 'W': W}})
    y, s, _ = hmm.get_samples(data_len)
    # --- plotter
    plotter(y, s, hmm, 1)
    return y


def update(y, n_states):
    data_dim, data_len = y.shape
    # --- setting
    hmm = Hmm(data_dim, n_states, expt_init_mode='random')
    mu = zeros((data_dim, n_states))
    hmm.set_params({'MuR': {'mu': mu}})
    hmm.init_expt_s(data_len)
    # --- plotter
    s = hmm.expt_s.argmax(0)
    plotter(y, s, hmm, 2)
    # --- update
    hmm.update(y)
    # --- plotter
    s = hmm.expt_s.argmax(0)
    plotter(y, s, hmm, 3)


def main():
    from matplotlib import pyplot as plt
    data_dim = 3
    n_states = 3
    data_len = 2000
    y = gen_data(data_dim, n_states, data_len)
    update(y, n_states)
    plt.show()


if __name__ == '__main__':
    main()
