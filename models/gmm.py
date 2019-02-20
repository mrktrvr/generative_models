#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
gmm.py
'''
import os
import sys
import cPickle as pickle

from numpy import sum as nsum
from numpy import newaxis
from numpy import zeros
from numpy import exp
from numpy import einsum
from numpy.random import choice
from numpy.random import multivariate_normal as mvnrand

from model_util import ModelUtil
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
        if self.prior is not None:
            self.prior.clear_params()
            self.prior = None
        if self.post is not None:
            self.post.clear_params()
            self.post = None

    def update(self, Y, S, YY=None):
        '''
        qmur.update(Y, S, YY=None)
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
        ln_lkh -= 0.5 * einsum('dek,det->kt', self.post.expt_prec, YY)
        ln_lkh += einsum('dt,dk->kt', Y, self.post.expt_prec_mu)
        ln_lkh -= 0.5 * self.post.expt_mu_prec_mu[:, newaxis]
        ln_lkh += 0.5 * self.post.expt_lndet_prec[:, newaxis]
        # ln_lkh -= 0.5 * data_dim * log(2 * np_pi)
        return ln_lkh

    def calc_kl_divergence(self):
        kl_norm_wish = 0
        '''
        for k in xrange(self.n_states):
            kl_norm_wish += KL_Norm_Wish(
                self.post.beta[k], self.post.mu[:, k], self.post.nu[k],
                self.post.W[:, :, k], self.prior.beta[k],
                self.prior.mu[:, k], self.prior.nu[k],
                self.prior.W[:, :, k])
        '''
        return kl_norm_wish

    def get_samples(self, data_len, by_posterior=True):
        if by_posterior:
            mu, R = self.post.sample_mu_R(data_len)
        else:
            mu, R = self.prior.sample_mu_R(data_len)
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
        qpi = qPi(n_states)
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
        # kl_dir = KL_Dir(self.post.ln_alpha, self.prior.ln_alpha)
        kl_dir = 0
        return kl_dir

    def get_samples(self, data_len=1, by_posterior=True):
        '''
        qpi.get_samples(data_len=1, by_posterior=True)
        @argvs
        data_len: data_lengs
        by_posterior: sample from posterior(True) or prior(False)
        '''
        if by_posterior:
            alpha = self.post.sample(data_len)
        else:
            alpha = self.prior.sample(data_len)
        return alpha

    def get_param_dict(self, by_posterior=True):
        '''
        qpi.get_param_dict(by_posterior=True)
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


class Theta():
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
        self.update_order = ['mu_R', 'pi']

    def set_default_params(self):
        '''
        theta.set_default_params()
        '''
        self.qmur.set_default_params()
        self.qpi.set_default_params()

    def clear_params(self):
        '''
        theta.clear_params()
        '''
        self.qmur.clear_params()
        self.qpi.clear_params()

    def update(self, Y, expt_S, YY=None):
        '''
        theta.update(Y, expt_S, YY=None)
        @argv
        Y: data, np.array(data_dim, data_len)
        expt_S: <S>, np.array(n_states, data_len)
        YY: YY^T, np.array(data_dim, data_dim, data_len)
        '''
        for ut in self.update_order:
            if ut == 'mu_R':
                self.qmur.update(Y, expt_S, YY=YY)
            elif ut == 'pi':
                self.qpi.update(expt_S)
            else:
                logger.error('%s is not supported' % ut)

    def get_params(self, by_posterior=True):
        '''
        theta.get_params(by_posterior)
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

    def save_param_dict(self, file_name):
        '''
        theta.save_param_dict(file_name)
        @argvs
        file_name: string
        '''
        prm = self.get_post_params()
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
        if by_posterior:
            mu, R = self.qmur.post.sample_mu_R()
            pi = self.qpi.post.sample()
        else:
            mu, R = self.qmur.prior.sample_mu_R()
            pi = self.qpi.prior.sample()
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
        expt_init_mode: initialisation mode, 'random' or 'kmeans'
        '''
        self.n_states = n_states
        self.expt_init_mode = argvs.get('expt_init_mode', 'random')
        self.expt = None
        self.const = 0
        self._eps = 1e-10
        self.updated = False

    def init_expt(self, data_len, **argvs):
        '''
        qS.init_expt(data_len, argvs)
        @argvs
        data_len: int
        **argvs:
            Y: (data_dim, data_len)
            S: (data_len)
            qpi: qPi object
        expt: (n_states, data_lem)
        '''
        expt = ModelUtil.init_expt_s(data_len, self.n_states,
                                     self.expt_init_mode, **argvs)
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
        self.updated = True

    def get_samples(self, data_len, pi, return_uniform_s=False):
        '''
        qS.get_samples(data_len, pi, return_uniform_s=False)
        @argv
        data_len: sample data length, int
        pi: chass object, qPi()
        return_uniform_s: flag to return uniformly assigned S, default False
        @return
        S: sampled data, np.array(data_len)
        '''
        S = zeros(data_len, dtype=int)
        if return_uniform_s:
            state_len = int(data_len / float(self.n_states))
            t = state_len + data_len % self.n_states
            for k in xrange(1, self.n_states):
                S[t:(t + state_len)] = k
                t += state_len
        else:
            for t in xrange(data_len):
                k = choice(self.n_states, p=pi)
                S[t] = k
        return S

    def get_estims(self, Y, theta, force_update=True):
        '''
        qS.get_estims(Y, theta, force_update=True)
        @argvs
        Y: observation data, np.array(data_dim, data_len)
        theta: class object, Theta()
        force_update: return hidden vriables from cotained expt_s
                      whether the model was updated or not
        @return
        S: estimated hidden variables, np.array(data_len)
        '''
        if not self.updated or force_update:
            if Y is None:
                logger.warn('qS was not updated. return initial expt_s')
            else:
                data_len = Y.shape[-1]
                self.init_expt(data_len, Y=Y)
                self.update(Y, theta)
        S = self.expt.argmax(0)
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
        self.expt_init_mode = argvs.get('expt_init_mode', 'random')
        # self.expt_init_mode = argvs.get('expt_init_mode', 'kmeans')
        # --- classes
        self.theta = Theta(data_dim, n_states)
        self.qs = qS(n_states, expt_init_mode=self.expt_init_mode)
        # --- learning params
        self.update_order = argvs.get('update_order', ['E', 'M'])
        self.max_em_itr = argvs.get('max_em_itr', 5)
        # --- variational bounds
        self.vbs = zeros(self.max_em_itr)
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

    def init_expt_S(self, data_len, Y=None):
        if self.qs.expt is None:
            self.qs.init_expt(data_len, Y=Y, qpi=self.theta.qpi)

    def update(self, Y):
        '''
        update posteriors
        gmm.update(Y)
        @argv:
        Y: observation data, np.array(data_dim, data_len)
        '''
        self.data_dim, data_len = Y.shape
        self.init_expt_S(data_len, Y)
        for i in xrange(self.max_em_itr):
            self.log_info_update_itr(self.max_em_itr, i)
            for j, uo in enumerate(self.update_order):
                if uo == 'E':
                    self.qs.update(Y, self.theta)
                elif uo == 'M':
                    self.theta.update(Y, self.qs.expt)
                else:
                    logger.error('%s is not supported' % uo)
            self._update_vb(i)

    def _update_vb(self, i):
        '''
        update variational bound
        '''
        self.vbs[i] = self.calc_vb()
        if not self.check_vb_increase(self.vbs, i):
            logger.error('vb increased')
            # embed(header='calc_vb')

    def calc_vb(self):
        '''
        calculate variational bound
        @return
        vb: float
        '''
        kl_mur = self.theta.qmur.calc_kl_divergence()
        kl_pi = self.theta.qpi.calc_kl_divergence()
        vb = self.qs.const - kl_pi - kl_mur
        return vb

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
        return_uniform_s = argvs.get('return_uniform_s', False)
        by_posterior = argvs.get('by_posterior', True)
        mu, R, pi = self.theta.get_samples(by_posterior)
        S = self.qs.get_samples(data_len, pi, return_uniform_s)
        Y = zeros((self.data_dim, data_len))
        for t in xrange(data_len):
            k = S[t]
            cov = inv(R[:, :, k])
            Y[:, t] = mvnrand(mu[:, k], cov)
        return Y, S, [mu, R, pi]

    def get_estims(self, Y=None, use_sample=False):
        '''
        gmm.get_estims(Y=None, use_sample=False)
        @argv
        Y: np.array(data_dim, data_len)
        use_sample: flag to use sampled estimation values or not, default False
        @return
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

    def save_params(self, file_name):
        was_saved = self.theta.save_param_dict(file_name)
        logger.info('saved %s' % file_name)
        return was_saved

    def load_params(self, file_name):
        was_loaded = self.theta.load_param_dict(file_name)
        self.data_dim = self.theta.data_dim
        self.n_states = self.theta.n_states
        return was_loaded


def gen_data(data_dim, n_states, data_len):
    print data_dim, n_states, data_len
    gmm = Gmm(data_dim, n_states, expt_init_mode='random')
    gmm.set_default_params()
    Y, S, prm = gmm.get_samples(data_len)
    return Y, S


def run_gmm(Y, S, n_states):
    from eval_disaggregation import EvalDisagg
    print '%d states GMM' % n_states
    data_dim, data_len = Y.shape
    gmm = Gmm(data_dim, n_states, expt_init_mode='random')
    gmm.set_default_params()
    gmm.update(Y)
    res_S = gmm.qS.expt.argmax(0)


def main_just_run():
    from IPython import embed
    data_dim = 2
    n_states = 5
    data_len = 10000
    Y, S = gen_data(data_dim, n_states, data_len)
    run_gmm(Y, S, 2)
    run_gmm(Y, S, 3)
    run_gmm(Y, S, 4)
    embed(header=__name__)


def main_example():
    from numpy.random import randn
    data_dim = 2
    n_states = 5
    data_len = 10000
    # --- data
    gmm = Gmm(data_dim, n_states, expt_init_mode='random')
    gmm.set_default_params()
    mu = randn(data_dim, n_states)
    gmm.set_params({'MuR': {'mu': mu}})
    Y, S, prm = gmm.get_samples(data_len)
    # --- training
    data_dim, data_len = Y.shape
    gmm = Gmm(data_dim, n_states, expt_init_mode='random')
    gmm.set_default_params()
    gmm.update(Y)
    # --- result
    Y, S, prm = gmm.get_samples(data_len)


if __name__ == '__main__':
    # main_just_run()
    main_example()
