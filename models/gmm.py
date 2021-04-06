#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
gmm.py
'''
import os
import sys
# import cPickle as pickle
import pickle

from numpy import newaxis
from numpy import nan
from numpy import atleast_2d
from numpy import zeros
from numpy import ones
from numpy import exp
from numpy import sum as nsum
from numpy import einsum
from numpy.random import choice
from numpy.random import dirichlet
from numpy.random import multivariate_normal as mvnrand

from model_util import CheckTools

cdir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(cdir, '..'))
from distributions.normal_wishart import NormalWishart
from distributions.dirichlet import Dirichlet
from util.calc_util import inv
from util.calc_util import logsumexp
from util.logger import logger


class qMuR():
    def __init__(self, data_dim, n_states, do_set_prm=False):
        '''
        mur = qMuR(data_dim, n_states)
        '''
        self.data_dim = data_dim
        self.n_states = n_states
        self.prior = None
        self.post = None
        if do_set_prm:
            self.set_default_param()

    def set_default_params(self):
        self.prior = NormalWishart(self.data_dim, self.n_states, True)
        self.post = NormalWishart(self.data_dim, self.n_states, True)

    def set_params(self, **argvs):
        '''
        mur.set_params(argvs)
        @args
        mu: data_dim x n_states
        beta: n_states
        nu: n_states, must be more than data_dim
        W: data_dim x data_dim x n_states
        inv_W: data_dim x data_dim x n_states
        '''
        n_states = argvs.get('mu', zeros((0, 0))).shape[-1]
        if n_states == 0:
            n_states = len(argvs.get('beta', []))
        if n_states == 0:
            n_states = len(argvs.get('nu', []))
        self.n_states = n_states if n_states != 0 else self.n_states
        self.prior = NormalWishart(self.data_dim, self.n_states)
        self.prior.set_params(**argvs)
        self.data_dim, self.n_states = self.prior.mu.shape
        self.post = NormalWishart(self.data_dim, self.n_states)
        self.post.set_params(**argvs)

    def clear_params(self):
        '''
        mur.cler_params()
        '''
        if self.prior is not None:
            self.prior.clear_params()
            self.prior = None
        if self.post is not None:
            self.post.clear_params()
            self.post = None

    def update(self, Y, S, YY=None):
        '''
        mur.update(Y, S, YY=None)
        Y: data_dim x data_len
        S: n_states x data_len
        prior: NormalWishart
        post: NormalWishart
        '''
        sum_S = nsum(S, 1)
        # --- beta
        beta_h = self.prior.beta + sum_S
        # --- nu
        nu_h = self.prior.nu + sum_S
        # --- mu
        sum_SY = einsum('kt,dt->dk', S, Y)
        beta_m = einsum('k,dk->dk', self.prior.beta, self.prior.mu)
        mu_h = einsum('dk,k->dk', sum_SY + beta_m, 1.0 / beta_h)
        # --- W
        if YY is None:
            sum_SYY = einsum('kt,dt,et->dek', S, Y, Y)
        else:
            sum_SYY = einsum('kt,det->dek', S, YY)
        bmm = einsum('dk,ek->dek', beta_m, self.prior.mu)
        bmmh = einsum('k,dk,ek->dek', beta_h, mu_h, mu_h)
        inv_W_h = sum_SYY + bmm - bmmh + self.prior.inv_W
        self.post.set_params(beta=beta_h, nu=nu_h, mu=mu_h, inv_W=inv_W_h)

    def calc_ln_lkh(self, Y, YY=None):
        '''
        calc_ln_lkh(Y, YY=None)
        @argv
        Y: observation data, np.array(data_dim, data_len)
        YY: Y Y^T, np.array(data_dim, data_dim, data_len), default None
        -/frac{1}{2} \{y_t^T <R^{(k)}> y_t
                       - 2 y_t^T <R^{(k)}\mu^{(k)}>
                       + <\mu^{(k)^T} R \mu^{(k)}>
                       + <\ln |R^{(k)}|>\}
        Y^T <R> Y = Tr[<R> Y Y^T]
        @ return
        ln_lkh: n_states x data_len
        '''
        data_dim, data_len = Y.shape
        ln_lkh = zeros((self.n_states, data_len))
        if YY is None:
            YY = einsum('dt,et->det', Y, Y)
        ln_lkh += einsum('dek,det->kt', self.post.expt_prec, YY)
        ln_lkh -= 2.0 * einsum('dt,dk->kt', Y, self.post.expt_prec_mu)
        ln_lkh += self.post.expt_mu_prec_mu[:, newaxis]
        ln_lkh -= self.post.expt_lndet_prec[:, newaxis]
        ln_lkh *= -0.5
        return ln_lkh

    def calc_kl_divergence(self):
        from distributions.kl_divergence import KL_Norm_Wish
        kl_norm_wish = 0
        for k in range(self.n_states):
            kl_norm_wish += KL_Norm_Wish(
                self.post.beta[k], self.post.mu[:, k], self.post.nu[k],
                self.post.W[:, :, k], self.prior.beta[k], self.prior.mu[:, k],
                self.prior.nu[k], self.prior.W[:, :, k])
        return kl_norm_wish[0]

    def get_samples(self, data_len=1, by_posterior=True):
        '''
        mu, R = mur.get_samples(data_len=1, by_posterior=True)
        '''
        if by_posterior:
            mu, R = self.post.sample_mu_R(data_len)
        else:
            mu, R = self.prior.sample_mu_R(data_len)
        return mu, R

    def get_expt(self, by_posterior=True):
        '''
        mu, R = mur.get_expt(by_posterior=True)
        '''
        if by_posterior:
            mu = self.post.mu
            R = self.post.expt_prec
        else:
            mu = self.prior.mu
            R = self.prior.expt_prec
        return mu, R

    def get_param_dict(self, by_posterior=True):
        if by_posterior:
            prm = self.post.get_param_dict()
        else:
            prm = self.prior.get_param_dict()
        return prm


class qPi(object):
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
        @argvs
        n_states: number of states
        '''
        self.n_states = n_states
        self.prior = None
        self.post = None

    def set_default_params(self):
        '''
        pi.set_default_params()
        '''
        self.prior = Dirichlet(self.n_states, do_set_prm=True)
        self.post = Dirichlet(self.n_states, do_set_prm=True)

    def set_params(self, **argvs):
        '''
        pi.set_params(**argvs)
        @argvs
        argvs:
            'alpha': n_states
            'ln_alpha': n_states
        '''
        self.prior = Dirichlet(self.n_states)
        self.prior.set_params(**argvs)
        self.post = Dirichlet(self.n_states)
        self.post.set_params(**argvs)

    def clear_params(self):
        '''
        pi.clear_params()
        '''
        if self.prior is not None:
            self.prior.clear_params()
        self.prior = None
        if self.post is not None:
            self.post.clear_params()
        self.post = None

    def update(self, expt_s):
        '''
        pi.update(expt_s)
        @argvs
        expt_s: np.array(n_states, data_len)
        '''

        alpha_h = self.prior.alpha + expt_s.sum(1)
        self.post.set_params(alpha=alpha_h)

    def calc_kl_divergence(self):
        '''
        @return
        kl_dir: KL divergence of Dirichlet distributions
        '''
        from distributions.kl_divergence import KL_Dir
        kl_dir = KL_Dir(self.post.ln_alpha, self.prior.ln_alpha)
        return kl_dir

    def get_param_dict(self, by_posterior=True):
        '''
        pi.get_param_dict(by_posterior=True)
        @argvs
        data_len: data_lengs
        by_posterior: parameters of posterior(True) or prior(False)
        @return
        dst: {'alpha': np.array(n_states), 'ln_alpha': np.array(n_states)}
        '''
        if by_posterior:
            dst = self.post.get_param_dict()
        else:
            dst = self.prior.get_param_dict()
        return dst

    def get_samples(self, data_len=1, by_posterior=True):
        '''
        pi.get_samples(data_len=1, by_posterior=True)
        @argvs
        data_len: data_lengs
        by_posterior: sample from posterior(True) or prior(False)
        '''
        if by_posterior:
            alpha = self.post.sample(data_len)
        else:
            alpha = self.prior.sample(data_len)
        return alpha

    def get_expt(self, by_posterior=True):
        '''
        pi.get_expt(by_posterior=True)
        @argvs
        by_posterior: from posterior(True) or prior(False)
        '''
        if by_posterior:
            pi = self.post.expt
        else:
            pi = self.prior.expt
        return pi


class Theta(object):
    '''
    theta = Theta(data_dim, n_states)
    '''

    def __init__(self, data_dim, n_states):
        '''
        theta = Theta(data_dim, n_states)
        @argvs
        data_dim: data dim, int
        n_states: number of states
        '''
        self.data_dim = data_dim
        self.n_states = n_states
        self.qmur = qMuR(self.data_dim, self.n_states)
        self.qpi = qPi(self.n_states)
        self.update_order = ['MuR', 'pi']

    def set_default_params(self):
        '''
        theta.set_default_params()
        '''
        self.qmur.set_default_params()
        self.qpi.set_default_params()

    def set_params(self, prm):
        '''
        theta.set_params(prm)
        @argvs
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
        '''
        if 'MuR' in prm:
            self.qmur.set_params(**prm['MuR'])
        else:
            if self.qmur.prior is None or self.qmur.post is None:
                self.qmur.set_default_params()
        if 'Pi' in prm:
            self.qpi.set_params(**prm['Pi'])
        else:
            if self.qpi.prior is None or self.qpi.post is None:
                self.qpi.set_default_params()
        self.n_states = self.qmur.n_states

    def clear_params(self):
        '''
        theta.clear_params()
        '''
        self.qmur.clear_params()
        self.qpi.clear_params()

    def update(self, Y, expt_s, YY=None):
        '''
        theta.update(Y, expt_s, YY=None)
        @argv
        Y: data, np.array(data_dim, data_len)
        expt_s: <S>, np.array(n_states, data_len)
        YY: YY^T, np.array(data_dim, data_dim, data_len)
        '''
        for ut in self.update_order:
            if ut == 'MuR':
                self.qmur.update(Y, expt_s, YY=YY)
            elif ut == 'pi':
                self.qpi.update(expt_s)
            else:
                logger.error('%s is not supported' % ut)

    def get_param_dict(self, by_posterior=True):
        '''
        theta.get_param_dict(by_posterior)
        @argvs
        by_posterior: parameters of posterior(True) or prior(False)
        @return
        dst: dictionary
            {'MuR':
             {'mu', 'beta', 'nu', 'W', 'inv_W',
              'expt_prec', 'expt_ln_det_prec',
              'expt_prec_mu', 'expt_mu_prec_mu'},
             'Pi': {'alpha', 'ln_alpha'}
            }
        '''
        dst = {
            'MuR': self.qmur.get_param_dict(by_posterior),
            'Pi': self.qpi.get_param_dict(by_posterior),
        }
        return dst

    def save_param_dict(self, file_name, by_posterior=True):
        '''
        theta.save_param_dict(file_name)
        @argvs
        file_name: string
        '''
        prm = self.get_param_dict(by_posterior)
        with open(file_name, 'w') as f:
            pickle.dump(prm, f)

    def load_param_dict(self, file_name):
        '''
        theta.load_param_dict(file_name)
        @argvs
        file_name: string
        '''
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                prm = pickle.load(f)
            self.set_params(prm)
            return True
        else:
            logger.warn('%s does not exist' % file_name)
            return False

    def get_samples(self, by_posterior=True):
        '''
        theta.get_samples()
        @argv
        by_posterior: flag to sample from posterior or prior
        '''
        mu, R = self.qmur.get_samples(by_posterior=by_posterior)
        pi = self.pi.get_samples(by_posterior=by_posterior)
        return mu, R, pi

    def get_expt(self, by_posterior=True):
        '''
        theta.get_expt()
        @argv
        by_posterior: flag to sample from posterior or prior
        '''
        mu, R = self.qmur.get_expt(by_posterior)
        pi = self.qpi.get_expt(by_posterior)
        return mu, R, pi


class qS(object):
    '''
    qs = qS(n_states)
    qs.expt: n_states x data_len
    qs.const: float
    '''

    def __init__(self, n_states, **argvs):
        '''
        qs = qS(n_states, expt_init_mode='random')
        qs = qS(n_states, expt_init_mode='kmeans')
        @argv
        n_states: number of states
        '''
        self.n_states = n_states
        self.expt = None
        self.const = 0
        self._eps = 1e-10

    def init_expt(self, data_len):
        '''
        qS.init_expt(data_len, argvs)
        @argvs
        data_len: int
        @self
        expt: (n_states, data_lem)
        '''
        alpha_pi = ones(self.n_states)
        expt = dirichlet(alpha_pi, size=data_len).T
        self.set_expt(expt)

    def set_expt(self, expt):
        '''
        qS.set_expt(expt)
        @argv
        expt: expectation of S, np.array(n_states, data_len)
        '''
        self.expt = expt

    def clear_expt(self):
        '''
        clear_expt()
        '''
        self.expt = None

    def update(self, Y, theta, YY=None):
        '''
        qS.update(Y, theta, YY=None)
        @argv
        Y: observation data, np.array(data_dim, data_len)
        theta: class object, Theta()
        YY: Y Y^T, np.array(data_dim, data_dim, data_len)
        '''
        ln_lkh_gmm = theta.qmur.calc_ln_lkh(Y, YY)
        ln_lkh = ln_lkh_gmm + theta.qpi.post.expt_ln[:, newaxis]
        norm = logsumexp(ln_lkh, 0)
        self.expt = exp(ln_lkh - norm[newaxis, :])
        self.const = norm.sum()

    def get_samples(self, data_len, pi=None):
        '''
        qS.get_samples(data_len, pi)
        @argv
        data_len: sample data length, int
        pi: np.array(n_states), probability array
        @return
        S: sampled data, np.array(data_len)
        '''
        S = zeros(data_len, dtype=int)
        if pi is None:
            pi = ones(self.n_states, dtype=float) / self.n_states
        for t in range(data_len):
            k = choice(self.n_states, p=pi)
            S[t] = k
        return S


class Gmm(CheckTools):
    '''
    gaussian mixture model
    gmm = Gmm(data_dim, n_states)
    gmm.set_default_params()
    gmm.set_params(prm)
    gmm.update(Y)
    # Y: np.array(data_dim, data_len)
    # data_dim: data dim
    # n_states: number of states
    # expt_s: expectations of hidden states. (n_states x data_len)
    '''

    def __init__(self, data_dim, n_states, **argvs):
        '''
        data_dim: data dim
        n_states: number of states
        '''
        self.data_dim = data_dim
        self.n_states = n_states
        # --- classes
        self.theta = Theta(data_dim, n_states)
        self.qs = qS(n_states)
        self.expt_s = self.qs.expt
        # --- learning params
        self.update_order = argvs.get('update_order', ['E', 'M'])
        self.max_em_itr = argvs.get('max_em_itr', 100)
        # --- variational bounds
        self.vbs = ones(self.max_em_itr) * nan
        self.do_eval = False

    def set_default_params(self):
        self.theta.set_default_params()

    def set_params(self, prm):
        '''
        @argv
        prm: dictcionary
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
        '''
        self.theta.set_params(prm)
        self.data_dim = self.theta.data_dim

    def init_expt_s(self, data_len):
        '''
        gmm.set_expt_s(expt, expt2=None)
        '''
        if self.qs.expt is None:
            self.qs.init_expt(data_len)
            self.expt_s = self.qs.expt

    def set_expt_s(self, expt):
        '''
        gmm.set_expt_s(expt)
        '''
        self.qs.set_expt(expt)
        self.expt_s = self.qs.expt

    def update(self, Y):
        '''
        gmm.update(Y)
        update posteriors
        @argv:
        Y: observation data, np.array(data_dim, data_len)
        '''
        self.data_dim, data_len = Y.shape
        self.init_expt_s(data_len)
        for i in range(self.max_em_itr):
            self.log_info_update_itr(self.max_em_itr, i)
            for j, uo in enumerate(self.update_order):
                if uo == 'E':
                    self.qs.update(Y, self.theta)
                elif uo == 'M':
                    self.theta.update(Y, self.qs.expt)
                else:
                    logger.error('%s is not supported' % uo)
            self._update_vb(i)
        self.expt_s = self.qs.expt

    def _update_vb(self, i):
        '''
        update variational bound
        '''
        vb, kl_mur, kl_pi = self.calc_vb()
        self.vbs[i] = vb
        logstr = (
            'const:%f, KL[MuR]: %f, KL[Pi]:%f vb: %f <- %f(prev vb)' %
            (self.qs.const, kl_mur, kl_pi, vb, self.vbs[i - 1]))
        logger.debug(logstr)
        if not self.check_vb_increase(self.vbs, i):
            logger.error('vb increased')

    def calc_vb(self):
        '''
        calculate variational bound
        @return
        vb: float
        '''
        kl_mur = self.theta.qmur.calc_kl_divergence()
        kl_pi = self.theta.qpi.calc_kl_divergence()
        vb = self.qs.const - kl_pi - kl_mur
        return vb, kl_pi, kl_mur

    def get_samples(self, data_len, **argvs):
        '''
        gmm.get_samples(data_len, use_uniform_s=False)
        get sampled data from the model
        @argv
        data_len: sample data length, int
        return_uniform_s: return uniform S or sampled S, boo, default, False
        by_posterior: sample from post or prior, bool, default True
        @return
        Y: sampled observation, np.array(data_dim, data_len)
        S: sampled hidden variables, np.array(data_len)
        mu: sampled mu, np.array(data_dim, n_states)
        R: sampled R, np.array(data_dim, data_dim, n_states)
        pi: sampled pi, np.array(n_states)
        '''
        by_posterior = argvs.get('by_posterior', True)
        # mu, R, pi = self.theta.get_samples(by_posterior)
        mu, R, pi = self.theta.get_expt(by_posterior)
        S = self.qs.get_samples(data_len, pi)
        Y = zeros((self.data_dim, data_len))
        cov = inv(R.transpose(2, 0, 1)).transpose(1, 2, 0)
        for t in range(data_len):
            k = S[t]
            Y[:, t] = mvnrand(mu[:, k], cov[:, :, k])
        return Y, S, [mu, R, pi]

    def save_params(self, file_name):
        was_saved = self.theta.save_param_dict(file_name)
        logger.info('saved %s' % file_name)
        return was_saved

    def load_params(self, file_name):
        was_loaded = self.theta.load_param_dict(file_name)
        self.data_dim = self.theta.data_dim
        self.n_states = self.theta.n_states
        return was_loaded


def plotter(y, s, gmm, figno=1):
    daat_dim, data_len = y.shape
    _, _, prm = gmm.get_samples(data_len)
    vbs = gmm.vbs
    # --- sample
    if False:
        _plotter_core(y, s, prm, vbs, 'sample', 100 * figno + 1)
    # --- expectation
    if True:
        prm = [
            gmm.theta.qmur.post.mu,
            gmm.theta.qmur.post.expt_prec,
            gmm.theta.qpi.post.expt,
        ]
        _plotter_core(y, s, prm, vbs, 'expextation', 100 * figno + 2)


def _plotter_core(y, s, prms, vbs, prm_type_str, figno):
    from util.plot_models import PlotModels
    mu, r, pi = prms
    n_cols = 3
    pm = PlotModels(3, n_cols, figno)
    idx1, idx2 = 0, 1
    # --- params
    pm.plot_2d_array((0, 0), mu, title=r'$\mu$ %s' % prm_type_str)
    pm.multi_bar((0, 1), atleast_2d(pi), title=r'$\pi$ %s' % prm_type_str)
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
    from util.calc_util import rand_wishart
    from numpy.random import randn
    gmm = Gmm(data_dim, n_states, expt_init_mode='random')
    gmm.set_default_params()
    mu = randn(data_dim, n_states) * 10
    W = rand_wishart(data_dim, n_states, c=1, by_eye=True)
    gmm.set_params({'MuR': {'mu': mu, 'W': W}})
    y, s, _ = gmm.get_samples(data_len)
    # --- plotter
    plotter(y, s, gmm, 1)
    return y


def update(y, n_states):
    data_dim, data_len = y.shape
    # --- setting
    gmm = Gmm(data_dim, n_states, expt_init_mode='random')
    # mu = zeros((data_dim, n_states))
    # gmm.set_params({'MuR': {'mu': mu}})
    gmm.init_expt_s(data_len)
    # --- plotter
    s = gmm.expt_s.argmax(0)
    # plotter(y, s, gmm, 2)
    # --- update
    gmm.update(y)
    # --- plotter
    s = gmm.expt_s.argmax(0)
    plotter(y, s, gmm, 3)


def main():
    from matplotlib import pyplot as plt
    data_dim = 2
    n_states = 3
    data_len = 2000
    y = gen_data(data_dim, n_states, data_len)
    update(y, n_states)
    plt.pause(1)


if __name__ == '__main__':
    main()
