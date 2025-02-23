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
from numpy import atleast_2d
from numpy import append
from numpy import zeros
from numpy import ones
from numpy import exp
from numpy import sum as nsum
from numpy import einsum
from numpy.random import choice
from numpy.random import multivariate_normal as mvnrand

cdir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(cdir)
from model_util import init_expt
from model_util import CheckTools

sys.path.append(os.path.join(cdir, '..'))
from distributions.normal_wishart import NormalWishart
from distributions.dirichlet import Dirichlet
from distributions.kl_divergence import KL_Norm_Wish
from distributions.kl_divergence import KL_Dir
from utils.calc_utils import inv
from utils.calc_utils import logsumexp
from utils.logger import logger

from matplotlib import pyplot as plt
from IPython import embed


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
            self.set_default_params()

    def set_default_params(self):
        self.prior = NormalWishart(self.data_dim, self.n_states, True)
        prms = self.prior.get_params()
        self.post = NormalWishart(self.data_dim, self.n_states, False)
        self.post.set_params(**prms)

    def set_params(self, **args):
        '''
        mur.set_params(args)
        @args
        mu: data_dim x n_states
        beta: n_states
        nu: n_states, must be more than data_dim
        W: data_dim x data_dim x n_states
        inv_W: data_dim x data_dim x n_states
        '''
        n_states = args.get('mu', zeros((0, 0))).shape[-1]
        if n_states == 0:
            n_states = len(args.get('beta', []))
        if n_states == 0:
            n_states = len(args.get('nu', []))
        self.n_states = n_states if n_states != 0 else self.n_states
        self.prior = NormalWishart(self.data_dim, self.n_states)
        self.prior.set_params(**args)
        self.data_dim, self.n_states = self.prior.mu.shape
        self.post = NormalWishart(self.data_dim, self.n_states)
        self.post.set_params(**args)

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
        kl_norm_wish = 0
        for k in range(self.n_states):
            kl_norm_wish += KL_Norm_Wish(self.post.beta[k], self.post.mu[:, k],
                                         self.post.nu[k], self.post.W[:, :, k],
                                         self.prior.beta[k], self.prior.mu[:,
                                                                           k],
                                         self.prior.nu[k], self.prior.W[:, :,
                                                                        k])
        return kl_norm_wish[0]

    def sample_mu(self, data_len=1, by_posterior=True):
        if by_posterior:
            R = self.post.expt_prec
            mu = self.post.sample_mu(data_len, R=R)
        else:
            R = self.prior.expt_prec
            mu = self.prior.sample_mu(data_len, R=R)

        return mu

    def samples(self, data_len=1, by_posterior=True):
        '''
        mu, R = mur.samples(data_len=1, by_posterior=True)
        '''
        if by_posterior:
            mu, R = self.post.sample_mu_R(data_len)
        else:
            mu, R = self.prior.sample_mu_R(data_len)
        return mu, R

    def expectations(self, by_posterior=True):
        '''
        mu, R = mur.expectations(by_posterior=True)
        '''
        if by_posterior:
            mu = self.post.mu
            R = self.post.expt_prec
        else:
            mu = self.prior.mu
            R = self.prior.expt_prec
        return mu, R

    def get_params(self, by_posterior=True):
        if by_posterior:
            prm = self.post.get_params()
        else:
            prm = self.prior.get_params()
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
        @args
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

    def set_params(self, **args):
        '''
        pi.set_params(**args)
        @args
        args:
            'alpha': n_states
            'ln_alpha': n_states
        '''
        self.prior = Dirichlet(self.n_states)
        self.prior.set_params(**args)
        self.post = Dirichlet(self.n_states)
        self.post.set_params(**args)

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
        @args
        expt_s: np.array(n_states, data_len)
        '''

        alpha_h = self.prior.alpha + expt_s.sum(1)
        self.post.set_params(alpha=alpha_h)

    def calc_kl_divergence(self):
        '''
        @return
        kl_dir: KL divergence of Dirichlet distributions
        '''
        kl_dir = KL_Dir(self.post.ln_alpha, self.prior.ln_alpha)
        return kl_dir

    def get_params(self, by_posterior=True):
        '''
        pi.get_params(by_posterior=True)
        @args
        data_len: data_lengs
        by_posterior: parameters of posterior(True) or prior(False)
        @return
        dst: {'alpha': np.array(n_states), 'ln_alpha': np.array(n_states)}
        '''
        if by_posterior:
            dst = self.post.get_params()
        else:
            dst = self.prior.get_params()
        return dst

    def samples(self, data_len=1, by_posterior=True):
        '''
        pi.samples(data_len=1, by_posterior=True)
        @args
        data_len: data_lengs
        by_posterior: sample from posterior(True) or prior(False)
        '''
        if by_posterior:
            alpha = self.post.sample(data_len)
        else:
            alpha = self.prior.sample(data_len)
        return alpha

    def expectations(self, by_posterior=True):
        '''
        pi.expectations(by_posterior=True)
        @args
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
        @args
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

    def get_params(self, by_posterior=True):
        '''
        theta.get_params(by_posterior)
        @args
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
            'MuR': self.qmur.get_params(by_posterior),
            'Pi': self.qpi.get_params(by_posterior),
        }
        return dst

    def save_params(self, file_name, by_posterior=True):
        '''
        theta.save_params(file_name)
        @args
        file_name: string
        '''
        prm = self.get_params(by_posterior)
        with open(file_name, 'w') as f:
            pickle.dump(prm, f)

    def load_params(self, file_name):
        '''
        theta.load_params(file_name)
        @args
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

    def samples(self, by_posterior=True):
        '''
        theta.samples()
        @argv
        by_posterior: flag to sample from posterior or prior
        '''
        mu, R = self.qmur.samples(by_posterior=by_posterior)
        pi = self.qpi.samples(by_posterior=by_posterior)
        return mu, R, pi

    def expectations(self, by_posterior=True):
        '''
        theta.expectations()
        @argv
        by_posterior: flag to sample from posterior or prior
        '''
        mu, R = self.qmur.expectations(by_posterior)
        pi = self.qpi.expectations(by_posterior)
        return mu, R, pi


class qS(object):
    '''
    qs = qS(n_states)
    qs.expt: n_states x data_len
    qs.const: float
    '''

    def __init__(self, n_states, **args):
        '''
        qs = qS(n_states, expt_init_mode='random')
        qs = qS(n_states, expt_init_mode='kmeans')
        @argv
        n_states: number of states
        '''
        self.n_states = n_states
        self._expt_init_mode = args.get('expt_init_mode', 'random')
        self.expt = None
        self._ln_lkh = None
        self.const = 0
        self._eps = 1e-10

    def init_expt(self, data_len, obs=None):
        '''
        qS.init_expt(data_len, args)
        @args
        data_len: int
        @self
        expt: (n_states, data_lem)
        '''
        expt = init_expt(data_len, self.n_states, obs, self._expt_init_mode)
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
        self._ln_lkh = ln_lkh_gmm + theta.qpi.post.expt_ln[:, newaxis]
        norm = logsumexp(self._ln_lkh, 0)
        self.expt = exp(self._ln_lkh - norm[newaxis, :])
        self.const = norm.sum()

    def predict(self, Y, theta):
        '''
        qS.update(Y, theta, YY=None)
        @argv
        Y: observation data, np.array(data_dim, data_len)
        theta: class object, Theta()
        '''
        ln_lkh_gmm = theta.qmur.calc_ln_lkh(Y, None)
        self._ln_lkh = ln_lkh_gmm + theta.qpi.post.expt_ln[:, newaxis]
        norm = logsumexp(self._ln_lkh, 0)
        self.expt = exp(self._ln_lkh - norm[newaxis, :])
        self.const = norm.sum()
        return self.expt

    def samples(self, data_len, pi=None):
        '''
        qS.samples(data_len, pi)
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

    def __init__(self, data_dim, n_states, **args):
        '''
        data_dim: data dim
        n_states: number of states
        '''
        self.data_dim = data_dim
        self.n_states = n_states

        # --- learning params
        self.update_order = args.get('update_order', ['E', 'M'])
        self._expt_init_mode = args.get('expt_init_mode', 'random')
        self._vb_th_to_stop_itr = args.get('vb_th_to_stop_itr', 1e-7)
        self._do_early_stop = args.get('do_early_stop', True)

        # --- model classes
        self.theta = Theta(data_dim, n_states)
        self.qs = qS(n_states, expt_init_mode=self._expt_init_mode)
        self.expt_s = self.qs.expt

        # --- variational bounds
        self.expt_s = None
        self.vbs = None

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

    def init_expt_s(self, data_len, obs=None):
        '''
        gmm.init_expt_s(data_len, obs=None)
        '''
        if self.qs.expt is None:
            self.qs.init_expt(
                data_len,
                obs,
            )
            self.expt_s = self.qs.expt

    def set_expt_s(self, expt):
        '''
        gmm.set_expt_s(expt)
        '''
        self.qs.set_expt(expt)
        self.expt_s = self.qs.expt

    def update(self, Y, max_em_itr=20):
        '''
        gmm.update(Y)
        update posteriors
        @argv:
        Y: observation data, np.array(data_dim, data_len)
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
        # --- Y Y'
        YY = einsum('dt,et->det', Y, Y)
        # --- initialise expectation
        self.init_expt_s(data_len)
        # --- EM iteration
        logger.info('update order: %s' % self.update_order)
        for i in range(ibgn, iend):
            self.log_info_update_itr(iend, i, interval_digit=1)
            for j, uo in enumerate(self.update_order):
                if uo == 'E':
                    self.qs.update(Y, self.theta, YY)
                elif uo == 'M':
                    self.theta.update(Y, self.qs.expt, YY)
                else:
                    logger.error('%s is not supported' % uo)
                # self._temp_plot(Y, i, uo)
            is_conversed = self._update_vb(i)
            # --- early stop
            if is_conversed:
                break

        self.expt_s = self.qs.expt

    def _temp_plot(self, Y, i, uo):
        prm = [
            self.theta.qmur.post.mu,
            self.theta.qmur.post.expt_prec,
            self.theta.qpi.post.expt,
        ]
        S = self.qs.expt.argmax(0)
        title = 'itr %d %s' % (i, uo)
        vbs = self.vbs[:i + 1] if i == 0 else self.vbs[:i]
        _plotter_core(Y, S, prm, vbs, 'expextation', title, 100 * 10)
        plt.pause(1)

    def _update_vb(self, i):
        '''
        update variational bound
        '''
        vb, kl_mur, kl_pi = self.calc_vb()
        self.vbs[i] = vb
        logstr = ('const:%f, KL[MuR]: %f, KL[Pi]:%f vb: %f <- %f(prev vb)' %
                  (self.qs.const, kl_mur, kl_pi, vb, self.vbs[i - 1]))
        logger.debug(logstr)
        if not self.check_vb_increase(self.vbs, i):
            dst = False
        else:
            is_conversed = self.is_conversed(self.vbs, i,
                                             self._vb_th_to_stop_itr)
            if is_conversed and self._do_early_stop:
                self.vbs = self.vbs[:(i + 1)]
                dst = True
            else:
                dst = False
        return dst

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

    def predict(self, Y=None, use_sample=False):
        '''
        return
        predict_y: (data_dim, data_len), waveform sequence
        predict_s: (data_len), state sequence which contains 0 to n_states - 1
        vb: float value, valiational bound
        '''
        expt_s = self.qs.predict(Y, self.theta)
        self.expt_s = expt_s
        predict_s = expt_s.argmax(0)
        predict_y = zeros((self.data_dim, len(predict_s)))
        if use_sample:
            for k in range(self.n_states):
                idx = predict_s == k
                data_len = predict_y[:, idx].shape[-1]
                mu, R = self.theta.qmur.post.sample()
                predict_y[:, idx] = mvnrand(mu[:, k],
                                            inv(R[:, :, k]),
                                            size=data_len).T
        else:
            for k in range(self.n_states):
                idx = predict_s == k
                data_len = predict_y[:, idx].shape[-1]
                m = self.theta.qmur.post.mu[:, k]
                c = inv(self.theta.qmur.post.expt_prec[:, :, k])
                predict_y[:, idx] = mvnrand(m, c, size=data_len).T
        vb = self.calc_vb()
        return predict_y, predict_s, vb

    def samples(self, data_len, **args):
        '''
        gmm.samples(data_len, use_uniform_s=False)
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
        by_posterior = args.get('by_posterior', True)
        # mu, R, pi = self.theta.samples(by_posterior)
        mu, R, pi = self.theta.expectations(by_posterior)
        S = self.qs.samples(data_len, pi)
        Y = zeros((self.data_dim, data_len))
        cov = inv(R.transpose(2, 0, 1)).transpose(1, 2, 0)
        for t in range(data_len):
            k = S[t]
            Y[:, t] = mvnrand(mu[:, k], cov[:, :, k])
        return Y, S, [mu, R, pi]

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
        'n_states': int,
        'data_dim': int,
        }
        '''
        dst = self.theta.get_params(by_posterior)
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


def plotter(y, s, gmm, title, figno=1):
    daat_dim, data_len = y.shape
    _, _, prm = gmm.samples(data_len)
    vbs = gmm.vbs
    # --- sample
    if False:
        _plotter_core(y, s, prm, vbs, 'sample', title, 100 * figno + 1)
    # --- expectation
    if True:
        prm = [
            gmm.theta.qmur.post.mu,
            gmm.theta.qmur.post.expt_prec,
            gmm.theta.qpi.post.expt,
        ]
        _plotter_core(y, s, prm, vbs, 'expextation', title, 100 * figno + 2)


def _plotter_core(y, s, prms, vbs, prm_type_str, sup_title, figno):
    from ml_utils.plot_models import PlotModels
    mu, r, pi = prms
    n_cols = 3
    pm = PlotModels(3, n_cols, figno)
    # --- mu
    pm.plot_2d_array((0, 0), mu, title=r'$\mu$ %s' % prm_type_str)
    # --- Pi
    pm.multi_bar((0, 1), atleast_2d(pi), title=r'$\pi$ %s' % prm_type_str)
    # --- A
    # --- No A
    # --- Obs (data_dim v amplitude)
    pm.plot_2d_array((1, 0), y, title='Y')
    # --- mu and obs (Scatter)
    cov = inv(r.transpose(2, 0, 1)).transpose(1, 2, 0)
    pm.plot_2d_mu_cov((1, 1), mu, cov, src=y, cat=s)
    # --- Variational bound
    pm.plot_vb((1, 2), vbs, cspan=1)
    # --- Y (sequence)
    pm.plot_seq((2, 0), y, cat=s, title='Y', cspan=n_cols)
    # ---
    pm.sup_title(sup_title)
    pm.tight_layout()


def gen_data(data_dim, n_states, data_len):
    from ml_utils.calc_utils import rand_wishart
    from numpy.random import seed
    from numpy import arange
    from numpy import tile
    from numpy import eye
    from numpy import nonzero
    from numpy import concatenate
    from numpy import array as arr
    seed(1)
    gmm = Gmm(data_dim, n_states, expt_init_mode='random')
    gmm.set_default_params()
    if False:
        from numpy.random import randn
        mu = randn(data_dim, n_states) * 2
    else:
        from numpy import linspace
        mu = linspace(-1, 1, data_dim * n_states)
        # mu = exp(linspace(-1, 1, data_dim * n_states) * 10)
        mu = mu.reshape(n_states, data_dim).T
        # W = rand_wishart(data_dim, n_states, c=1e+3, by_eye=True)
        # W = tile(eye(data_dim) * 10, (n_states, 1, 1))
        w_diag = arange(n_states * data_dim).reshape(n_states, data_dim) + 1
        w_diag = w_diag[:, -1] * 1e+2
        W = arr([eye(data_dim) * w_diag[k] for k in range(n_states)])
        W = W.transpose(1, 2, 0)
    prms = {
        'MuR': {
            'mu': mu,
            'W': W,
        }
    }
    gmm.set_params(prms)
    y, s, _ = gmm.samples(data_len)
    idx = concatenate([nonzero(s == k)[0] for k in range(n_states)])
    y = y[:, idx]
    # --- plotter
    plotter(y, s, gmm, 'GMM data', 1)
    return y


def update(y, n_states):
    from numpy.random import randn
    from numpy import arange
    from numpy import eye
    from numpy import array as arr
    from numpy import cov
    from numpy.random import multivariate_normal as mvn_rand
    from sklearn.cluster import KMeans

    data_dim, data_len = y.shape

    # --- setting
    gmm_setting = {
        # 'update_order': ['M', 'E'],
        'update_order': ['E', 'M'],
        # 'expt_init_mode': 'random',
        'expt_init_mode': 'kmeans',
    }
    gmm = Gmm(data_dim, n_states, **gmm_setting)

    # --- Mu
    mu_mode = 'mvn_rand'
    if mu_mode == 'zeros':
        mu = zeros((data_dim, n_states))
    elif mu_mode == 'randn':
        c = 2
        mu = randn(data_dim, n_states) * c
    elif mu_mode == 'arange':
        mu = arange(data_dim * n_states).reshape(data_dim, n_states)
    elif mu_mode == 'kmeans':
        km = KMeans(n_states)
        km.fit(y.T)
        mu = km.cluster_centers_.T
    elif mu_mode == 'mvn_rand':
        mu = mvn_rand(y.mean(1), cov(y), size=n_states).T
    else:
        raise Exception('Not supported: %s' % mu_mode)

    alpha = ones(n_states) * (data_len / n_states)
    W = arr([eye(data_dim) * 1e+3 for k in range(n_states)])
    W = W.transpose(1, 2, 0)
    # --- set params
    prms = {'MuR': {'mu': mu, 'W': W}, 'Pi': {'alpha': alpha}}
    gmm.set_params(prms)
    gmm.init_expt_s(data_len, y)
    # --- plotter
    s = gmm.expt_s.argmax(0)
    plotter(y, s, gmm, 'GMM prior', 2)
    # --- update
    gmm.update(y, 200)
    # --- plotter
    s = gmm.expt_s.argmax(0)
    plotter(y, s, gmm, 'posterior', 3)

    predict_y, predict_s, vb = gmm.predict(y)


def main():
    from matplotlib import pyplot as plt
    data_dim = 2
    n_states = 3
    data_len = 100
    y = gen_data(data_dim, n_states, data_len)
    update(y, n_states)
    plt.ion()
    plt.show()
    input('return to finish')


if __name__ == '__main__':
    main()
