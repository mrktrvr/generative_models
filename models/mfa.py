#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
mfa.py
'''
import os
import sys
import pickle

from numpy import pi as np_pi
from numpy import nan
from numpy import newaxis
from numpy import eye
from numpy import zeros
from numpy import ones
from numpy import log
from numpy import exp
from numpy import tile
from numpy import einsum
from numpy.random import multivariate_normal as mvnrnd

from gmm import qPi
from gmm import qS as GmmS
from fa import qZ as FaZ
from fa import qLamb as FaLamb
from fa import qR as FaR

cdir = os.path.abspath(os.path.dirname(__file__))
lib_root = os.path.join(cdir, '..')
sys.path.append(lib_root)
from util.logger import logger
from util.calc_util import inv
from util.calc_util import logdet
from util.calc_util import logsumexp


class qZS(FaZ):
    '''
    z = qZS(fa_dim, n_states)
    z_mu: array(aug_dim, n_states, data_len)
    z_cov: array(aug_dim, aug_dim, n_states)
    expt_s: array(n_states, data_len)
    expt_zs: array(aug_dim, n_states, data_len)
    expt_szz: array(aug_dim, aug_dim, n_states, data_len)
    '''

    def __init__(self, fa_dim, n_states):
        super(qZS, self).__init__(fa_dim)
        self.n_states = n_states
        # ---
        self.s = qS(n_states)
        # --- prior of z
        self.prior = None
        self.set_default_priors()
        # --- posterior
        self.z_mu = None
        self.z_cov = None
        # --- expectation
        self.expt_s = None
        self.expt_sz = None
        self.expt_szz = None

    def init_expt(self, data_len):
        '''
        zs.init_expt(data_len)
        '''
        self.s.init_expt(data_len)
        m, c = self.prior.mu[:, 0], self.prior.cov[:, :, 0]
        n_states = self.n_states
        self.z_mu = mvnrnd(m, c, size=(data_len, n_states)).transpose(2, 1, 0)
        self.z_cov = tile(self.prior.cov, (n_states))
        self.set_expt(self.s.expt, self.z_mu, self.z_cov)

    def set_expt(self, expt_s, z_mu_h, z_cov_h):
        '''
        zs.set_expt(data_len):
        expt_s: array(n_states, data_len)
        z_mu_h: array(aug_dim, n_states, data_len)
        z_cov_h: array(aug_dim, aug_dim, n_states)
        '''
        self.expt_s = expt_s
        self.expt_sz = einsum('lkt,kt->lkt', z_mu_h, self.expt_s)
        zz = einsum('lkt,jkt->ljkt', self.expt_sz, self.expt_sz)
        expt_zz = zz + z_cov_h[:, :, :, newaxis]
        self.expt_szz = einsum('kt,ljkt->ljkt', expt_s, expt_zz)

    def update(self, Y, theta):
        '''
        @argvs
        Y: array(data_dim, data_len)
        lamb.mu: <lamb> array(aug_dim, data_dim)
        lamb.post.expt2: <lamblambT> array(aug_dim, aug_dim, data_dim)
        r.post.expt: <R> array(data_dim)
        lamb.post.mu: <lamb> array(aug_dim, data_dim)
        '''
        # -- prc
        rll = einsum('ddk,ljdk->ljk', theta.r.post.expt, theta.lamb.post.expt2)
        z_prc = self.prior.prec + rll
        # --- cov
        self.z_cov = inv(z_prc.transpose(2, 0, 1)).transpose(1, 2, 0)
        # --- mu
        ylr = einsum('dt,ldk,ddk->lkt', Y, theta.lamb.post.mu,
                     theta.r.post.expt)
        ylr_zmu = ylr + self.prior.expt_prec_mu[:, :, newaxis]
        self.z_mu = einsum('ljk,jkt->lkt', self.z_cov, ylr_zmu)
        # --- update s
        z_mpm = einsum('lkt,lkt->kt', self.z_mu, ylr_zmu)
        z_cov_lndet = logdet(self.z_cov.transpose(2, 0, 1))
        self.s.update(Y, theta, self.prior, z_mpm, z_cov_lndet)
        # --- expectations
        self.set_expt(self.s.expt, self.z_mu, self.z_cov)

    def get_samples(self, data_len, pi=None, by_posterior=True):
        '''
        zs.get_samples(data_len, pi=None, by_postrior=True)
        @argvs
        data_len: data length
        pi: array(n_states), probability, sum must be 1
        by_posterior: default True
        @returns
        z: array(aug_dim, n_states, data_len)
        s: array(n_states, data_len)
        '''
        # --- sample S
        s = self.s.get_samples(data_len, pi)
        # --- sample Z
        if by_posterior:
            z = ones((self.aug_dim, self.n_states, data_len)) * nan
            z[:-1, :, :] = 0
            z[-1, :, :] = 1
            mu_len = 0 if self.z_mu is None else self.z_mu.shape[-1]
            t_max = data_len if data_len < mu_len else mu_len
            for t in range(t_max):
                k = s[t]
                z[:, k, t] = mvnrnd(self.z_mu[:, k, t], self.z_cov[:, :, k])
            t_rest = data_len - t_max
            if t_rest > 0:
                z[:, :, t_max:] = self.prior.sample(t_rest).transpose(1, 2, 0)
        else:
            z = self.prior.sample(data_len)[:, :, 0].T
        return z, s


class qS(GmmS):
    '''
    S = qS(n_states)
    expt: array(n_states, data_len)
    '''

    def __init__(self, n_states):
        super(qS, self).__init__(n_states)

    def update(self, Y, theta, z_prior, zs_mpm, zs_cov_lndet):
        '''
        qS.update(Y, theta, z_prior, zs_mpm, zs_cov_lndet)
        @argv
        Y: observation data, np.array(data_dim, data_len)
        theta: class object, Theta()
        z_prior:
        zs_mpm:
        zs_cov_lndet:
        self.expt: array(n_states, data_len)
        '''
        data_dim, data_len = Y.shape
        ln_lkh = zeros((self.n_states, data_len))
        ln_lkh += einsum('dt,dt,ddk->kt', Y, Y, theta.r.post.expt)
        ln_lkh -= theta.r.post.expt_lndet[:, newaxis]
        ln_lkh += z_prior.expt_mu_prec_mu
        ln_lkh += z_prior.expt_lndet_prec
        ln_lkh += zs_cov_lndet[:, newaxis]
        ln_lkh += zs_mpm
        ln_lkh += data_dim * log(2 * np_pi)
        ln_lkh *= -0.5
        ln_lkh += theta.pi.post.ln_alpha[:, newaxis]
        norm = logsumexp(ln_lkh, axis=0)
        self.expt = exp(ln_lkh - norm[newaxis, :])
        self.const = norm.sum()


class qLamb(FaLamb):
    '''
    lamb = Lamb(fa_dim, data_dim, n_states)
    lamb.prior
    lamb.post
    '''

    def __init__(self, fa_dim, data_dim, n_states, do_set_prm=False):
        '''
        prior:
        post:
        mu: <Lamb> (aug_dim, data_dim, n_states)
        cov: cov<Lamb> (aug_dim, aug_dim, data_dim, n_states)
        prc: inv(cov<Lamb>) (aug_dim, aug_dim, data_dim, n_states)
        expt2: <LambLamb'> = <Lamb><Lamb>'+ cov
        '''
        super(qLamb, self).__init__(fa_dim, data_dim, False)
        self.n_states = n_states
        if do_set_prm:
            self.set_default_param()

    def update(self, r, sum_szz, sum_ysz):
        '''
        r.post.diag_expt: (data_dim, n_states)
        szz: array(aug_dim, aug_dim, n_states, data_len) YSS
        sum_ysz: array(data_dim, aug_dim, n_states) sum(Y<ZS>)
        '''
        # --- precision
        r_szz = einsum('ddk,ijk->ijdk', r.post.expt, sum_szz)
        prc = self.prior.prec + r_szz
        # --- covariance
        cov = inv(prc.transpose((2, 3, 0, 1))).transpose((2, 3, 0, 1))
        # --- mu
        rysz = einsum('ddk,dlk->ldk', r.post.expt, sum_ysz)
        pm_rysz = self.prior.expt_prec_mu + rysz
        mu = einsum('ijdk,jdk->idk', cov, pm_rysz)
        self.post.set_params(mu=mu, cov=cov, prc=prc)

    def get_samples(self, data_len=1, by_posterior=True):
        if by_posterior:
            m = self.post.sample(data_len)
        else:
            m = self.prior.sample(data_len)
        return m


class qR(FaR):
    '''
    r = R(data_dim, n_states)
    r.prior
    r.post
    a: array(data_dim, n_states)
    b: array(data_dim, n_states)
    '''

    def __init__(self, data_dim, n_states, do_set_prm=False):
        '''
        r = R(data_dim, n_states)
        data_dim: data dimensioin
        n_states: number of states
        '''
        super(qR, self).__init__(data_dim, False)
        self.n_states = n_states
        if do_set_prm:
            self.set_default_param()

    def update(self, lamb, s, sum_szz, sum_yys, sum_ysz):
        '''
        lamb.post.mu: array(aug_dim, data_dim, n_states)
        yys: array(data_dim, n_states, data_len)
        ysz: array(data_dim, aug_dim, n_states, data_len)
        s: array(n_states, data_len)
        szz: array(aug_dim, aug_dim, n_states, data_len)
        '''
        # --- a
        a = self.prior.a + 0.5 * einsum('kt->k', s)
        # --- b
        yszl = einsum('dlk,ldk->dk', sum_ysz, lamb.post.mu)
        tr_szzll = einsum('ljk,ljdk->dk', sum_szz, lamb.post.expt2)
        b = self.prior.b + 0.5 * (sum_yys - 2.0 * yszl + tr_szzll)
        self.post.set_params(a=a, b=b)

    def get_samples(self, data_len=1, by_posterior=True):
        if by_posterior:
            r = self.post.sample(data_len)
        else:
            r = self.prior.sample(data_len)
        r = einsum('dk,de->dek', r, eye(self.data_dim))
        return r


class Theta(object):
    '''
    theta = Theta(fa_dim, data_dim, n_states)
    '''

    def __init__(self, fa_dim, data_dim, n_states):
        '''
        theta = Theta(fa_dim, data_dim, n_states)
        '''
        self.fa_dim = fa_dim
        self.aug_dim = self.fa_dim + 1
        self.data_dim = data_dim
        self.n_states = n_states
        self.lamb = qLamb(fa_dim, data_dim, n_states)
        self.r = qR(data_dim, n_states)
        self.pi = qPi(n_states)
        self.update_order = ['Lamb', 'R', 'Pi']

    def set_default_params(self):
        '''
        theta.set_default_params()
        '''
        self.lamb.set_default_params()
        self.r.set_default_params()
        self.pi.set_default_params()

    def set_params(self, prm):
        '''
        theta.set_params(prm)
        @argvs
        prm: dictionary
        {'Lamb': {
            'mu':: (aug_gim, data_dim, n_states),
            'cov': (aug_dim, da_dim, data_dim, n_states),
            'prec': (aug_dim, da_dim, data_dim, n_states),
            },
         'R': {
            'a': (data_dim, n_states),
            'b': (data_dim, n_states)
            },
         'Pi': {
            'alpha': (n_states),
            'ln_alpha': (n_states)
            },
        '''
        if 'Lamb' in prm:
            self.lamb.set_params(**prm['Lamb'])
        else:
            if self.lamb.prior is None or self.lamb.post is None:
                self.lamb.set_default_params()
        if 'R' in prm:
            self.r.set_params(**prm['R'])
        else:
            if self.r.prior is None or self.r.post is None:
                self.r.set_default_params()
        if 'Pi' in prm:
            self.pi.set_params(**prm['Pi'])
        else:
            if self.pi.prior is None or self.pi.post is None:
                self.pi.set_default_params()
        self.fa_dim = self.lamb.fa_dim
        self.aug_dim = self.lamb.aug_dim
        self.data_dim = self.lamb.data_dim
        self.n_states = self.lamb.n_states

    def clear_params(self):
        '''
        theta.clear_params()
        '''
        self.lamb.clear_params()
        self.r.clear_params()

    def update(self, Y, zs):
        '''
        theta.update(Y, zs)
        Y: np.array(data_dim, data_len)
        zs: qZS class object
        '''
        if zs.mu is None:
            zs.init_expt(Y.shape[-1])
        s = zs.s.expt
        sum_szz = einsum('ljkt->ljk', zs.expt_szz)
        sum_ysz = einsum('dt,lkt->dlk', Y, zs.expt_sz)
        sum_yys = einsum('dt,dt,kt->dk', Y, Y, zs.s.expt)
        for uo in self.update_order:
            if uo == 'Lamb':
                self.lamb.update(self.r, sum_szz, sum_ysz)
            elif uo == 'R':
                self.r.update(self.lamb, s, sum_szz, sum_yys, sum_ysz)
            elif uo == 'Pi':
                self.pi.update(s)
            else:
                logger.error('%s is not supported' % uo)
                sys.exit(-1)

    def get_param_dict(self, by_posterior=True):
        '''
        theta.get_params(by_posterior)
        @argvs
        by_posterior: parameters of posterior(True) or prior(False)
        @return
        dst: dictionary
            {'Lamb': Lamb,
             'R': R,
             'Pi': Pi,
            }
        '''
        dst = {
            'Lamb': self.qlamb.get_param_dict(by_posterior),
            'R': self.qr.get_param_dict(by_posterior),
            'Pi': self.pi.get_param_dict(by_posterior),
        }
        return dst

    def get_samples(self, data_len=1, by_posterior=True):
        l = self.lamb.get_samples(data_len, by_posterior)
        r = self.r.get_samples(data_len, by_posterior)
        pi = self.pi.get_samples(data_len, by_posterior)
        return l, r, pi


class Mfa:
    '''
    mfa = Mfa(fa_dim, data_dim, n_states, n_states)
    '''

    def __init__(self, fa_dim, data_dim, n_states, **args):
        '''
        mfa = Mfa(fa_dim, data_dim, n_states, n_states)
        '''
        self.data_dim = data_dim
        self.fa_dim = fa_dim
        self.aug_dim = fa_dim + 1
        self.n_states = n_states
        # --- theta and zs
        self.zs = qZS(fa_dim, n_states)
        self.theta = Theta(self.fa_dim, self.data_dim, self.n_states)
        # --- update setting
        self.max_itr = args.get('max_itr', 100)
        self.update_order = [
            'Theta',
            'ZS',
        ]

    def init_zs(self, data_len):
        self.zs.init_expt(data_len)

    def set_default_params(self):
        self.theta.set_default_params()

    def set_params(self, prm):
        '''
        mfa.set_params(prm)
        @argvs
        prm: dictionary
        {'Lamb': {
            'mu':: (data_dim, n_states),
            'beta': (n_states)
            'nu': (n_states),
            'W': (data_dim, data_dim, n_states)
            },
         'R': {
            'a': (n_states),
            'b': (n_states)
            },
         'Pi': {
            'alpha': (n_states),
            'ln_alpha': (n_states)
            },
        }
        '''
        self.theta.set_params(prm)

    def update(self, Y):
        '''
        mfa.update()
        @argvs
        Y: np.array(data_dim, data_len)
        '''
        logger.info('update order %s, in Theta %s' % (self.update_order,
                                                      self.theta.update_order))
        for i in range(self.max_itr):
            for j, uo in enumerate(self.update_order):
                if uo == 'ZS':
                    self.zs.update(Y, self.theta)
                elif uo == 'Theta':
                    self.theta.update(Y, self.zs)
                else:
                    logger.error('%s is not supported' % uo)
                    sys.exit(-1)

    def get_param_dict(self, by_posterior=True):
        '''
        mfa.get_param_dict(by_posterior=True)
        '''
        dic = self.theta.get_param_dict(by_posterior)
        return dic

    def save_param_dict(self, file_name, by_posterior=True):
        '''
        mfa.save_param_dict(file_name, by_posterior=True)
        @argvs
        file_name: string
        '''
        dic = self.get_param_dict(by_posterior)
        dir_name = os.path.dirname(os.path.abspath(file_name))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(file_name, 'w') as f:
            pickle.dump(dic, f)

    def load_param_dict(self, file_name):
        '''
        mfa.save_param_dict(file_name)
        @argvs
        file_name: string
        '''
        ret = False
        try:
            with open(file_name, 'r') as f:
                prm = pickle.load(f)
            self.set_params(prm)
            ret = True
        except IOError as e:
            logger.error('%s' % e)
        return ret

    def get_samples(self, data_len, by_posterior=True):
        lamb, r, pi = self.theta.get_samples(by_posterior=by_posterior)
        z, s = self.zs.get_samples(data_len, pi, by_posterior)
        y = ones((self.data_dim, data_len)) * nan
        inv_r = inv(r.transpose(2, 0, 1)).transpose(1, 2, 0)
        for t in range(data_len):
            k = s[t]
            mu = einsum('ld,l->d', lamb[:, :, k], z[:, k, t])
            cov = inv_r[:, :, k]
            y[:, t] = mvnrnd(mu, cov)
        return y, z, s, [lamb, r, inv_r, pi]


def plotter(y, z, s, prms, title, figno=1):
    from numpy import diag
    from numpy import array as arr
    from util.plot_models import PlotModels
    l, r, inv_r, pi = prms
    aug_dim, dat_dim, n_states = l.shape
    n_cols = aug_dim + 1
    pm = PlotModels(3, n_cols, figno)
    # --- Lambda
    for i in range(aug_dim):
        pm.plot_2d_array((0, i), l[i, :, :], title='$\Lambda$[%d]' % i)
        pm.ax.legend(['%d' % x for x in range(n_states)], loc=0)
    # --- R
    tmp = arr([diag(r[:, :, i]) for i in range(n_states)]).T
    pm.plot_2d_array((0, n_cols - 1), tmp, title='R')
    pm.ax.legend(['%d' % x for x in range(n_states)], loc=0)
    # --- Y
    pm.plot_2d_array((1, 0), y, title='Y')
    pm.plot_seq((1, 1), y, cat=s, title='Y', cspan=n_cols - 1)
    # --- Z
    data_len = z.shape[-1]
    zs = arr([z[:, s[t], t] for t in range(data_len)]).T
    pm.plot_2d_array((2, 0), zs, title='Z')
    pm.plot_seq((2, 1), zs, cat=s, title='Z', cspan=n_cols - 1)
    pm.ax.legend(['%d' % x for x in range(z.shape[0])], loc=0)
    # ---
    pm.sup_title(title)
    pm.tight_layout()


def main():
    from matplotlib import pyplot as plt
    fa_dim = 3
    data_dim = 8
    n_states = 3
    data_len = 1000
    # --- data
    mfa = Mfa(fa_dim, data_dim, n_states)
    mfa.set_default_params()
    mfa.init_zs(data_len)
    Y, Z, S, prms = mfa.get_samples(data_len)
    plotter(Y, Z, S, prms, 'prior sample', 1)
    # --- update
    mfa = Mfa(fa_dim, data_dim, n_states)
    mfa.set_default_params()
    mfa.update(Y)
    Y, Z, S, prms = mfa.get_samples(data_len)
    plotter(Y, Z, S, prms, 'posterior sample', 2)
    plt.show()


if __name__ == '__main__':
    main()
