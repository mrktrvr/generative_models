#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
lda.py
'''
import os
import sys

import cPickle as pickle

from numpy import newaxis
from numpy import zeros
from numpy import ones
from numpy import array as arr
from numpy import einsum
from numpy import exp
from numpy import log
from numpy import unique
from numpy import concatenate
from numpy import histogram
from numpy import tile
from numpy.random import choice
from numpy.random import dirichlet

from gmm import qS as gmm_qs
from gmm import qPi as gmm_qpi
from model_util import CheckTools
from default_model_params import DefaultLdaParams

cdir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(cdir, '..'))
from distributions.dirichlet import Dirichlet
from util.calc_util import logsumexp
from util.logger import logger


def idx_seq_to_full_array(src, n_states):
    n_states = len(unique(src)) if n_states is None else n_states
    data_len = src.shape[-1]
    if src.ndim == 1:
        dst = zeros((n_states, data_len))
        for k in xrange(n_states):
            dst[k, src == k] = 1
    else:
        dst = src
    return dst


def calc_tf_idf(s_list, n_states):
    '''
    @argv
    s_list: list of HMM states
    n_staets: number of states of HMM
    @return
    tfidf: np.array(n_states, len(s_list))
    tf: np.array(n_states, len(s_list))
    idf: np.array(n_states)
    '''
    n_state_in_batch = arr(
        [[len(s[s == k]) for k in xrange(n_states)] for s in s_list])
    n_in_batch = arr([len(s) for s in s_list], dtype=float)
    tf = n_state_in_batch / n_in_batch[:, newaxis]
    tf = tf.T
    n_batches = len(s_list)
    n_batches_with_k = arr(
        [sum([1 if k in s else 0 for s in s_list]) for k in range(n_states)],
        dtype=float)
    n_batches_with_k += 1e-10
    idf = log(n_batches) - log(n_batches_with_k) + 1
    # tfidf = einsum('ij,i->ij', tf, idf)
    tfidf = tf * idf[:, newaxis]
    return tfidf, tf, idf


class qPi(object):
    '''
    qpi = qPi(n_states, n_cat)
    '''

    def __init__(self, n_states, n_cat):
        '''
        qpi = qPi(n_states, n_cat)
        n_states: number of states
        '''
        self.n_states = n_states
        self.n_cat = n_cat
        self.prior = None
        self.post = None

    def set_default_params(self):
        '''
        qpi.set_default_params()
        '''
        alpha = DefaultLdaParams().Pi(self.n_states, self.n_cat)
        self.set_params(alpha=alpha)

    def set_params(self, alpha, ln_alpha=None):
        '''
        qpi.set_params(**argvs)
        @argvs
        'alpha': np.array(n_states, n_cat)
        'ln_alpha': np.array(n_states, n_cat)
        '''
        self.n_states, self.n_cat = alpha.shape
        self.prior = Dirichlet(self.n_states, self.n_cat)
        self.post = Dirichlet(self.n_states, self.n_cat)
        argvs = {'alpha': alpha, 'ln_alpha': ln_alpha}
        self.prior.set_params(**argvs)
        self.post.set_params(**argvs)

    def clear_params(self):
        '''
        qpi.clear_params()
        '''
        if self.prior is not None:
            self.prior.clear_params()
        self.prior = None
        if self.post is not None:
            self.post.clear_params()
        self.post = None

    def update(self, S, expt_z):
        '''
        qpi.update(s, expt_z)
        @argv
        S: np.array(n_states, data_len), or np.array(data_len)
        expt_z: np.array(n_cat, data_len)
        @internal
        tmp: np.array(n_states, data_len)
        alpha_h: np.array(n_states, n_cat)
        '''
        tmp = idx_seq_to_full_array(S, self.n_states)
        sum_sz = einsum('kt,ct->kc', tmp, expt_z)
        alpha_h = self.prior.alpha + sum_sz
        self.post.set_params(alpha=alpha_h)

    def get_samples(self, data_len=1, by_posterior=True):
        '''
        qpi.get_samples(data_len=1, by_posterior=True)
        @argvs
        data_len: int
        by_posterior: sample from posterior(True) or prior(False)
        @return
        alpha: np.array(data_len)
        '''
        if by_posterior:
            alpha = self.post.sample(data_len)
        else:
            alpha = self.prior.sample(data_len)
        return alpha

    def get_param_dict(self, by_posterior=True):
        '''
        phi.get_param_dict(by_posterior=True)
        @argvs
        by_posterior: parameters of posterior(True) or prior(False)
        @return
        dst: {'alpha': np.array(n_cat), 'ln_alpha': np.array(n_cat)}
        '''
        if by_posterior:
            dst = self.post.get_param_dict()
        else:
            dst = self.prior.get_param_dict()
        return dst


class qPhi(gmm_qpi):
    def __init__(self, n_cat):
        '''
        n_cat: number of categories
        '''
        super(qPhi, self).__init__(n_cat)
        self.n_cat = n_cat

    def set_default_params(self):
        alpha = DefaultLdaParams().Phi(self.n_cat)
        self.set_params(alpha=alpha)

    def update(self, expt_z):
        '''
        qphi.update(expt_z)
        @argv
        expt_z: np.array(n_cat, data_len)
        @internal
        alpha_h: np.array(n_cat)
        '''
        alpha_h = expt_z.sum(1) + self.prior.alpha
        self.post.set_params(alpha=alpha_h)
        self.n_cat = self.n_states


class qZ(gmm_qs):
    '''
    qz = qZ(n_cat)
    '''

    def __init__(self, n_cat):
        '''
        qz = qz(n_cat)
        n_cat: number of categories, int
        '''
        super(qZ, self).__init__(n_cat)
        self.expt_init_mode = 'random'
        self.n_cat = n_cat

    def update(self, S, qpi, qphi, tfidf=None):
        '''
        qz.update(S, qpi, qphi)
        @argvs
        s: np.array(n_states, data_len)
        qpi: qpi object
        qphi: qphi object
        @internal
        expt_ln_pi: n_states x n_cat
        s_pi: n_cat x data_len
        xi: n_cat x data_len
        '''
        tmp = idx_seq_to_full_array(S, qpi.n_states)
        ln_pi = qpi.post.expt_ln
        if tfidf is not None:
            ln_pi += log(tfidf.sum(1) + 1e-23)[:, newaxis]
        s_pi = einsum('kt,kc->ct', tmp, ln_pi)
        xi = s_pi + qphi.post.expt_ln[:, newaxis]
        norm = logsumexp(xi, 0)
        self.expt = exp(xi - norm[newaxis, :])
        self.const = norm
        self.updated = True

    def init_expt(self, data_len):
        '''
        qz.init_expt(data_len, argvs)
        @argvs
        data_len: int
        @self
        expt: (n_states, data_lem)
        '''
        alpha_pi = ones(self.n_states)
        expt = dirichlet(alpha_pi, size=data_len).T
        self.set_expt(expt)

    def get_samples(self, data_len, phi):
        '''
        qz.get_samples(data_len , phi)
        @argvs
        data_len: int
        phi: np.array(n_cat)
        @output
        z: np.array(data_len)
        '''
        z = zeros(data_len, dtype=int)
        for t in xrange(data_len):
            z[t] = choice(self.n_cat, p=phi)
        return z

    def get_estims(self):
        '''
        qz.get_estims()
        '''
        pass


class Lda(CheckTools):
    '''
    '''

    def __init__(self, n_states, n_cat, **argvs):
        '''
        n_states: number of states, int
        n_cat: number of categories, int
        n_batches: number of batches, int
        '''
        self.n_states = n_states
        self.n_cat = n_cat
        self.max_em_itr = argvs.get('max_em_itr', 50)
        self.tfidf = argvs.get('tfidf', None)
        # --- distributions
        self.qpi = qPi(n_states, n_cat)
        self.qphi_list = None
        self.qz_list = None
        # --- update setting
        self.update_order_lda = ['Pi', 'Phi', 'Z']
        # self.update_order_lda = ['Z', 'Pi', 'Phi']
        # self.update_order_lda = ['Z', 'Phi', 'Pi']

    def init_qphi_qz(self, n_batches):
        self.init_qphi(n_batches)
        self.init_qz(n_batches)

    def init_qphi(self, n_batches):
        if self.qphi_list is None or len(self.qphi_list) != n_batches:
            self.qphi_list = [qPhi(self.n_cat) for b in xrange(n_batches)]

    def init_qz(self, n_batches):
        if self.qz_list is None or len(self.qz_list) != n_batches:
            self.qz_list = [qZ(self.n_cat) for b in xrange(n_batches)]

    def clear_qphi_qz(self):
        self.qphi_list = None
        self.qz_list = None

    def init_expt_z(self, data_len_list):
        self.init_qphi_qz(len(data_len_list))
        for b, data_len in enumerate(data_len_list):
            if self.qz_list[b].expt is None:
                self.qz_list[b].init_expt(data_len)

    def set_default_params(self, n_batches):
        self.qpi.set_default_params()
        self.init_qphi(n_batches)
        for b in xrange(n_batches):
            self.qphi_list[b].set_default_params()

    def set_params(self, prms):
        if 'Pi' in prms:
            self.qpi.set_params(**prms['Pi'])
        if 'Phi' in prms:
            n_batches = len(prms['Phi'])
            self.init_qphi(n_batches)
            for b, p in enumerate(prms['Phi']):
                self.qphi_list[b].set_params(**p)

    def _print_params(self, i, tar_list=['Phi', 'Z', 'Pi']):
        print '--- itr %d %s' % (i, tar_list)
        if 'Phi' in tar_list:
            print 'qphi post (n_batches x n_cat)'
            for b in xrange(self.n_batches):
                print self.qphi_list[b].post.expt
            print 'qphi prior (n_batches x n_cat)'
            for b in xrange(self.n_batches):
                print self.qphi_list[b].prior.expt
        if 'Z' in tar_list:
            print 'qz (n_batches x n_cat)'
            for b in xrange(self.n_batches):
                print self.qz_list[b].expt.sum(1)
        if 'Pi' in tar_list:
            print 'qpi.post(n_states x n_cat)\n', self.qpi.post.expt.T
            print 'qpi.prior(n_states x n_cat)\n', self.qpi.prior.expt.T
        print '---'

    def update(self, s_list):
        '''
        lda.update(s_list)
        s_list: list of np.array(n_states, data_len)
        '''
        self.update_lda(s_list)

    def update_lda(self, s_list):
        '''
        lda.update(s_list)
        s_list: list of np.array(n_states, data_len) or np.array(data_len)
        '''
        n_batches = len(s_list)
        self.init_qphi_qz(n_batches)
        # self._print_params(-1)
        self._update_core(s_list, self.update_order_lda, self.max_em_itr)

    def predict(self, s_list, n_itr=1):
        n_batches = len(s_list)
        self.init_qphi_qz(n_batches)
        # update_order = self.update_order_lda
        update_order = ['Z', 'Phi', 'Pi']
        # update_order = ['Z', 'Phi']
        self._update_core(s_list, update_order, n_itr)

    def _update_core(self, s_list, update_order, n_itr):
        logger.info('order:%s, itr: %d' % (update_order, n_itr))
        for i in xrange(n_itr):
            if i % 10 == 0:
                logger.info('itr %3d of %3d' % (i, n_itr))
            for uo in update_order:
                if uo == 'Phi':
                    for b in xrange(len(self.qphi_list)):
                        self.qphi_list[b].update(self.qz_list[b].expt)
                elif uo == 'Z':
                    for b in xrange(len(self.qz_list)):
                        self.qz_list[b].update(
                            s_list[b], self.qpi, self.qphi_list[b], self.tfidf)
                elif uo == 'Pi':
                    s = concatenate(s_list, 0 if s_list[0].ndim == 1 else 1)
                    z = concatenate([x.expt for x in self.qz_list], 1)
                    self.qpi.update(s, z)
                else:
                    raise Exception('update order %s does not exist' % uo)

    def get_samples(self, data_len_list, by_posterior=True):
        '''
        lda.get_samples(data_len_list, by_posterior=True)
        '''
        s = []
        z = []
        phi = []
        self.init_qphi_qz(len(data_len_list))
        pi = self.qpi.get_samples(1, by_posterior)
        for b, dl in enumerate(data_len_list):
            phi_b = self.qphi_list[b].get_samples(1, by_posterior)
            z_b = self.qz_list[b].get_samples(dl, phi_b)
            s_b = zeros(dl, dtype=int)
            for t in xrange(dl):
                s_b[t] = choice(self.n_states, p=pi[:, z_b[t]])
            s.append(s_b)
            z.append(z_b)
            phi.append(phi_b)
        prms = [pi, phi]
        return s, z, prms

    def get_estims(self):
        z_expt = [x.expt for x in self.qz_list]
        phi_expt = arr([x.post.expt for x in self.qphi_list])
        pi_expt = self.qpi.post.expt
        return z_expt, phi_expt, pi_expt

    def get_param_dict(self, by_posterior=True):
        '''
        theta.get_param_dict(by_posterior=True)
        @argvs
        by_posterior: use posterior params(True) or not(False)
        @return
        dst = {
        'Pi': {
            'alpha': (n_states),
            'ln_alpha': (n_states),
            },
        'n_states': int,
        'n_cat': int,
        'data_dim': int,
        }
        '''
        dst = {
            'Pi': self.qpi.get_param_dict(by_posterior),
            'Phi': [x.get_param_dict() for x in self.qphi_list]
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


def plotter(s_list, z_list, prms, figno, msg):
    from util.plot_models import PlotModels
    logger.info('plotting %s %s' % (str(figno), msg))
    pi = prms[0]
    phi = arr(prms[1])
    pm = PlotModels(4, 2, figno)
    pm.multi_bar((0, 0), pi, title=r'$\pi$', xlbl='cat each state')
    pm.multi_bar((0, 1), phi, title=r'$\phi$', xlbl='cat each batch')
    n_states, n_cat = pi.shape
    # --- s hist
    s_seq = [histogram(x, bins=range(n_states + 1))[0] for x in s_list]
    s_seq = arr(s_seq).astype(float)
    s_seq = s_seq / s_seq.sum(1)[:, newaxis]
    ymax = s_seq.max()
    pm.plot_states_indv((1, 0), s_seq.T, ymax=ymax, title='S', cspan=1)
    # --- s_seq
    s_seq = concatenate(s_list, 0)
    pm.plot_states_indv((1, 1), s_seq, title='S', cspan=1)
    # --- z_hist
    z_seq = [histogram(x, bins=range(n_cat + 1))[0] for x in z_list]
    z_seq = arr(z_seq).astype(float)
    z_seq = z_seq / z_seq.sum(1)[:, newaxis]
    ymax = z_seq.max()
    pm.plot_states_indv((2, 0), z_seq.T, ymax=ymax, title='Z', cspan=1)
    # --- z_seq
    z_seq = concatenate(z_list, 0)
    pm.plot_states_indv((2, 1), z_seq, title='Z', cspan=1)
    tmp = [tile(prms[1][b], (x.shape[0], 1)).T for b, x in enumerate(s_list)]
    phi_seq = concatenate(tmp, 1)
    pm.plot_states_stack((3, 0), phi_seq, title='Phi', cspan=1)
    pm.plot_states_indv((3, 1), phi_seq, title='Phi', cspan=1)


def gen_data(n_batches=32, n_states=8, n_cat=4, do_plot=True, do_print=False):
    from numpy.random import randint
    data_len_list = [180] * n_batches
    alpha_pi = ones((n_states, n_cat))
    for k in xrange(n_states):
        c = randint(n_cat)
        alpha_pi[k, c] += 10
    alpha_phi = ones((n_batches, n_cat))
    for b in xrange(n_batches):
        c = randint(n_cat)
        alpha_phi[b, c] += 10
        c = randint(n_cat)
        alpha_phi[b, c] += 10
    phi_list = [{'alpha': x} for x in alpha_phi]
    prms = {
        'Pi': {
            'alpha': alpha_pi
        },
        'Phi': phi_list,
    }
    lda = Lda(n_states, n_cat)
    lda.set_default_params(n_batches)
    lda.set_params(prms)
    s_list, z_list, prms = lda.get_samples(data_len_list)
    if do_print:
        print 's'
        for b in xrange(n_batches):
            print idx_seq_to_full_array(s_list[b], n_states).sum(1)
        print 'z'
        for b in xrange(n_batches):
            print idx_seq_to_full_array(z_list[b], n_cat).sum(1)
    return s_list, n_states, n_cat, z_list, prms


def main():
    from matplotlib import pyplot as plt
    # --- data
    s_list, n_states, n_cat, z_list, prms = gen_data()
    plotter(s_list, z_list, prms, '01_data', 'data')
    n_batches = len(s_list)
    data_len_list = [len(x) for x in s_list]
    # --- model
    lda = Lda(n_states, n_cat)
    # --- initialisation
    lda.set_default_params(n_batches)
    lda.init_expt_z(data_len_list)
    # --- prior sample
    _, z_list_pri, prms_pri = lda.get_samples(data_len_list)
    plotter(s_list, z_list_pri, prms_pri, '00_init', 'before')
    # --- update
    lda.update(s_list)
    # --- posterior sample
    _, z_list_pst, prms_pst = lda.get_samples(data_len_list)
    plotter(s_list, z_list_pst, prms_pst, '02_estimate', 'after')
    plt.show()


if __name__ == '__main__':
    main()
