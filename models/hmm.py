#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
hmm.py
'''
import os
import sys
import pickle

from numpy import atleast_2d
from numpy import append
from numpy import zeros
from numpy import exp
from numpy import einsum
from numpy.random import choice
from numpy.random import multivariate_normal as mvnrand

from gmm import Theta as GmmTheta
from gmm import qPi as GmmPi
from gmm import qS as GmmS
from forward_backward import ForwardBackward
from model_util import CheckTools
from model_util import init_expt

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))
from distributions.dirichlet import Dirichlet
from distributions.kl_divergence import KL_Dir
from ml_utils.calc_utils import inv
from python_utilities.utils.logger import logger


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
        for k in range(self.n_states):
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

    def samples(self, data_len=1, by_posterior=True):
        '''
        theta.samples(data_len=1, by_posterior=True)
        @return
        mu:
        R:
        pi:
        A:
        '''
        mu, R = self.qmur.samples(data_len, by_posterior)
        pi = self.qpi.samples(data_len, by_posterior)
        A = self.qA.samples(data_len, by_posterior)
        return mu, R, pi, A

    def expectations(self, by_posterior=True):
        mu, R = self.qmur.expectations(by_posterior)
        pi = self.qpi.expectations(by_posterior)
        A = self.qA.expectations(by_posterior)
        return mu, R, pi, A

    def get_params(self, by_posterior=True):
        '''
        theta.get_params(by_posterior=True)
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
            'MuR': self.qmur.get_params(by_posterior),
            'Pi': self.qpi.get_params(by_posterior),
            'A': self.qA.get_params(by_posterior),
            'n_states': self.n_states,
            'data_dim': self.data_dim,
        }
        return dst

    def save_params(self, file_name, by_posterior=True):
        prm = self.get_params(by_posterior)
        try:
            with open(file_name, 'w') as f:
                pickle.dump(prm, f)
            return True
        except Exception as exception:
            logger.warning(exception)
            return False

    def load_params(self, file_name):
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
        super(qS, self).__init__(n_states, **argvs)
        self.expt2 = None

    def init_expt(self, data_len, obs=None):
        '''
        qs.init_expt(data_len)
        @argvs
        data_len: int
        @self
        expt: (n_states, data_lem)
        '''
        expt = init_expt(data_len, self.n_states, obs, self._expt_init_mode)
        # alpha_pi = ones(self.n_states)
        # alpha_A = ones((self.n_states, self.n_states))
        # expt = ones((self.n_states, data_len)) * nan
        # expt[:, 0] = dirichlet(alpha_pi)
        # for t in range(1, data_len):
        #     k_prev = choice(self.n_states, p=expt[:, t - 1])
        #     expt[:, t] = dirichlet(alpha_A[:, k_prev])
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
        ln_obs_lkh = theta.qmur.calc_ln_lkh(Y, YY)
        logger.debug('forward backward')
        lns, lnss, c = ForwardBackward()(ln_obs_lkh, theta.qpi.post.expt_ln,
                                         theta.qA.post.expt_ln)
        logger.debug('expt')
        s = exp(lns)
        ss = exp(lnss)
        self.expt = s
        self.expt2 = ss
        self.const = c

    def estimate(self, Y, theta):
        logger.debug('calc ln_lkh')
        ln_obs_lkh = theta.qmur.calc_ln_lkh(Y, None)
        logger.debug('forward backward')
        lns, lnss, c = ForwardBackward()(ln_obs_lkh, theta.qpi.post.expt_ln,
                                         theta.qA.post.expt_ln)
        expt = exp(lns)
        s = expt.argmax(0)
        return s

    def samples(self, data_len, pi, A):
        '''
        qS.samples(data_len, pi, A, use_uniform_s=False)
        @argvs
        data_len: int
        pi: np.array(n_states)
        A: np.array(n_states, n_states)
        '''
        S = zeros(data_len, dtype=int)
        for t in range(data_len):
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
    hmm.samples(data_len)
    hmm.init_expt_s(data_le)
    hmm.update(Y)
    '''

    def __init__(self, data_dim, n_states, **argvs):
        '''
        '''
        # --- data dimension and number of states
        self.data_dim = data_dim
        self.n_states = n_states

        # --- learning params
        self.update_order = argvs.get('update_order', ['E', 'M'])
        self._expt_init_mode = argvs.get('expt_init_mode', 'random')
        self._vb_th_to_stop_itr = argvs.get('vb_th_to_stop_itr', 1e-5)

        # --- model classes
        self.theta = Theta(data_dim, n_states)
        self.qs = qS(n_states, expt_init_mode=self._expt_init_mode)

        # --- data
        self.expt_s = None
        self.vbs = None

    def init_expt_s(self, data_len, obs=None):
        '''
        hmm.init_expt_s(data_len)
        '''
        if self.qs.expt is None:
            self.qs.init_expt(data_len, obs)
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

    def update(self, Y, max_em_itr=20):
        '''
        hmm.update(Y)
        '''
        # --- Index and array for VB
        if self.vbs is None:
            self.vbs = zeros(max_em_itr)
            ibgn = 0
        else:
            ibgn = len(self.vbs)
            self.vbs = append(self.vbs, zeros(max_em_itr))
        iend = ibgn + max_em_itr
        # --- data size
        self.data_dim, data_len = Y.shape
        # --- initialise expectation
        self.init_expt_s(data_len)
        # --- Y Y'
        YY = einsum('dt,et->det', Y, Y)
        # --- EM iteration
        logger.info('Update order: %s' % self.update_order)
        for i in range(ibgn, iend):
            self.log_info_update_itr(iend, i, interval_digit=1)
            for j, uo in enumerate(self.update_order):
                if uo == 'E':
                    self.qs.update(Y, self.theta, YY)
                elif uo == 'M':
                    self.theta.update(Y, self.qs.expt, self.qs.expt2, YY)
                else:
                    logger.error('%s is not supported' % uo)
            do_stop_itr = self._update_vb(i)
            # --- early stop
            if do_stop_itr:
                break
        self.expt_s = self.qs.expt

    def _update_vb(self, i):
        self.vbs[i] = self.calc_vb()
        if not self.check_vb_increase(self.vbs, i):
            logger.warning('vb decreased')
            dst = False
        else:
            if self.is_conversed(self.vbs, i, self._vb_th_to_stop_itr):
                dst = True
            else:
                dst = False
        if dst is True:
            self.vbs = self.vbs[:(i + 1)]
        return dst

    def calc_vb(self):
        kl_mur = self.theta.qmur.calc_kl_divergence()
        kl_pi = self.theta.qpi.calc_kl_divergence()
        kl_A = self.theta.qA.calc_kl_divergence()
        vb = self.qs.const - kl_pi - kl_A - kl_mur
        return vb

    def samples(self, data_len, by_posterior=True):
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
        # mu, R, pi, A = self.theta.samples(1, by_posterior)
        mu, R, pi, A = self.theta.expectations(by_posterior)
        S = self.qs.samples(data_len, pi, A)
        Y = zeros((self.data_dim, data_len))
        for t in range(data_len):
            k = S[t]
            cov = inv(R[:, :, k])
            Y[:, t] = mvnrand(mu[:, k], cov)
        return Y, S, [mu, R, pi, A]

    def estimate(self, Y=None, use_sample=False):
        '''
        return
        estim_y: (data_dim, data_len), waveform sequence
        estim_s: (data_len), state sequence which contains 0 to n_states - 1
        vb: float value, valiational bound
        '''
        estim_s = self.qs.estimate(Y, self.theta)
        estim_y = zeros((self.data_dim, len(estim_s)))
        if use_sample:
            for k in range(self.n_states):
                idx = estim_s == k
                data_len = estim_y[:, idx].shape[-1]
                mu, R = self.theta.qmur.post.sample()
                estim_y[:, idx] = mvnrand(
                    mu[:, k], inv(R[:, :, k]), size=data_len).T
        else:
            for k in range(self.n_states):
                idx = estim_s == k
                data_len = estim_y[:, idx].shape[-1]
                m = self.theta.qmur.post.mu[:, k]
                c = inv(self.theta.qmur.post.expt_prec[:, :, k])
                estim_y[:, idx] = mvnrand(m, c, size=data_len).T
        vb = self.calc_vb()
        return estim_y, estim_s, vb

    def get_params(self, by_posterior=True):
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
        dst = self.theta.get_params(by_posterior)
        # self.data_dim = dst['data_dim']
        return dst

    def save_params(self, file_name):
        was_saved = self.theta.save_params(file_name)
        logger.info('saved %s' % file_name)
        return was_saved

    def load_params(self, file_name):
        was_loaded = self.theta.load_params(file_name)
        self.data_dim = self.theta.data_dim
        self.n_states = self.theta.n_states
        return was_loaded

    def save_expt_s(self, file_name):
        self.qs.save_estims(file_name)

    def load_expt_s(self, file_name):
        dst = self.qs.load_estims(file_name)
        self.expt_s = self.qs.expt
        return dst


# def plotter(y, s, hmm, sup_title, figno):
#     daat_dim, data_len = y.shape
#     # --- sample
#     if False:
#         _, _, prm = hmm.samples(data_len)
#         vbs = hmm.vbs
#         cur_figno = 100 * figno + 1
#         _plotter_core(y, s, prm, vbs, 'sample', sup_title, cur_figno)
#     # --- expectation
#     if True:
#         prm = [
#             hmm.theta.qmur.post.mu,
#             hmm.theta.qmur.post.expt_prec,
#             hmm.theta.qpi.post.expt,
#             hmm.theta.qA.post.expt,
#         ]
#         vbs = hmm.vbs
#         cur_figno = 100 * figno + 2
#         _plotter_core(y, s, prm, vbs, 'expextation', sup_title, cur_figno)
#

# def _plotter_core(y, s, prm, vbs, prm_type_str, sup_title, figno):
#     from helpers.plot_models import PlotModels
#     mu, r, pi, A = prm
#     n_cols = 3
#     pm = PlotModels(3, n_cols, figno)
#     # --- Mu
#     pm.plot_2d_array((0, 0), mu, title=r'$\mu$ %s' % prm_type_str)
#     # --- Pi
#     pm.multi_bar((0, 1), atleast_2d(pi), title=r'$\pi$ %s' % prm_type_str)
#     # --- A
#     pm.plot_table((0, 2), A, title='Transition probabilities')
#     # --- Obs(data_dim v amplitude)
#     pm.plot_2d_array((1, 0), y, title='Y')
#     # --- Mu and obs
#     cov = inv(r.transpose(2, 0, 1)).transpose(1, 2, 0)
#     pm.plot_2d_mu_cov((1, 1), mu, cov, src=y, cat=s)
#     # --- Variational bound
#     pm.plot_vb((1, 2), vbs, cspan=1)
#     # --- Obs(sequence)
#     pm.plot_seq((2, 0), y, cat=s, title='Y', cspan=n_cols)
#     # ---
#     pm.sup_title(sup_title)
#     pm.tight_layout()


def gen_data(data_dim, n_states, data_len):
    from ml_utils.calc_utils import rand_wishart
    from numpy.random import seed
    seed(1)
    hmm = Hmm(data_dim, n_states, expt_init_mode='random')
    hmm.set_default_params()
    if False:
        from numpy.random import randn
        mu = randn(data_dim, n_states) * 2
    else:
        from numpy import linspace
        mu = linspace(-1, 1, data_dim * n_states) * 2
        mu = mu.reshape(n_states, data_dim).T
    W = rand_wishart(data_dim, n_states, c=1e+3, by_eye=True)
    hmm.set_params({'MuR': {'mu': mu, 'W': W}})
    y, s, _ = hmm.samples(data_len)
    # --- plotter
    # plotter(y, s, hmm, 'HMM data', 1)
    return y


def update(y, n_states):
    from numpy.random import randn
    data_dim, data_len = y.shape
    # --- setting
    uo = ['M', 'E']
    hmm = Hmm(data_dim, n_states, expt_init_mode='kmeans', update_order=uo)
    # --- Mu
    # mu = randn(data_dim, n_states) * 2
    # mu = zeros((data_dim, n_states))
    mu = randn(data_dim, n_states)
    # --- set params
    hmm.set_params({'MuR': {'mu': mu}})
    hmm.init_expt_s(data_len, y)
    # --- plotter
    s = hmm.expt_s.argmax(0)
    # plotter(y, s, hmm, 'HMM prior', 2)
    # --- update
    hmm.update(y, 100)
    # --- plotter
    s = hmm.expt_s.argmax(0)
    # plotter(y, s, hmm, 'HMM posterior', 3)


def main():
    from matplotlib import pyplot as plt
    data_dim = 2
    n_states = 4
    data_len = 2000
    y = gen_data(data_dim, n_states, data_len)
    update(y, n_states)
    plt.ion()
    plt.show()
    input('Return to finish')


if __name__ == '__main__':
    main()
