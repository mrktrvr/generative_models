#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
fa.py
'''
import os
import sys
import copy

from numpy import newaxis
from numpy import nan
from numpy import copy as ncopy
from numpy import zeros
from numpy import ones
from numpy import tile
from numpy import eye
from numpy import sum as nsum
from numpy import einsum
from numpy.random import multivariate_normal as mvnrnd

import pickle

CDIR = os.path.abspath(os.path.dirname(__file__))

sys.path.append(CDIR)
from default_model_params import DefaultFaParams

LIB_ROOT = os.path.join(CDIR, '..')
sys.path.append(LIB_ROOT)

from util.logger import logger
from util.calc_util import inv
from distributions.multivariate_normal import MultivariateNormal
from distributions.gamma import Gamma


class qZ(object):
    '''
    z = qZ(data_len, fa_dim)
    '''

    def __init__(self, fa_dim):
        '''
        '''
        # ---
        self.fa_dim = fa_dim
        self.aug_dim = fa_dim + 1
        # --- prior
        self.prior = None
        self.set_default_priors()
        self.mu = None
        self.cov = None
        self.prec = None
        self.expt2 = None

    def init_expt(self, data_len):
        '''
        self.mu: (aug_dim, data_len)
        self.cov: (aug_dim, aug_dim)
        self.prec: (aug_dim, aug_dim)
        self.expt2: (aug_dim, aug_dim, data_len)
        '''
        self.mu = tile(self.prior.mu, (1, data_len))
        self.cov = ncopy(self.prior.cov[:, :, 0])
        self.prec = ncopy(self.prior.prec[:, :, 0])
        self.expt2 = self.calc_expt2()

    def set_default_priors(self):
        m, c, p = DefaultFaParams().Z(self.fa_dim)
        m = m[:, newaxis]
        c = c[:, :, newaxis]
        p = p[:, :, newaxis]
        self.set_prior(mu=m, cov=c, prec=p)

    def set_prior(self, **argvs):
        self.prior = MultivariateNormal(self.aug_dim, 1)
        self.prior.set_params(**argvs)

    def update(self, theta, Y):
        '''
        z.update(Y)
        lamb.mu: <lamb> array(aug_dim, data_dim)
        lamb.post.expt2: <lamblambT> array(aug_dim, aug_dim, data_dim)
        r.post.expt: <R> array(data_dim)
        lamb.post.mu: <lamb> array(aug_dim, data_dim)

        '''
        # -- prec
        rll = einsum('ddk,ljdk->lj', theta.r.post.expt, theta.lamb.post.expt2)
        self.prec = rll + self.prior.prec[:, :, 0]
        # --- cov
        self.cov = inv(self.prec)
        # --- mean
        lr = einsum('ldk,ddk->ld', theta.lamb.post.mu, theta.r.post.expt)
        lry = einsum('ld,dt->lt', lr, Y)
        lry_muz = lry + self.prior.expt_prec_mu[:, 0, newaxis]
        self.mu = einsum('lj,jt->lt', self.cov, lry_muz)
        self.expt2 = self.calc_expt2()

    def samples(self, data_len, by_posterior=True):
        '''
        z.samples(data_len)
        data_len: 100
        @return
        sample_z: (aug_dim, n_states, data_len)
        '''
        if by_posterior:
            sample_z = ones((self.aug_dim, data_len)) * nan
            mu_len = 0 if self.mu is None else self.mu.shape[-1]
            t_max = data_len if data_len < mu_len else mu_len
            for t in range(t_max):
                sample_z[:, t] = mvnrnd(self.mu[:, t], self.cov)
            t_rest = data_len - t_max
            if t_rest > 0:
                sample_z[:, t_max:] = self.prior.sample(t_rest)[:, :, 0].T
        else:
            sample_z = self.prior.sample(data_len)[:, :, 0].T
        return sample_z

    def calc_expt2(self):
        mm = einsum('lt,jt->ljt', self.mu, self.mu) + self.cov[:, :, newaxis]
        return mm


class qLamb(object):
    '''
    lamb = qLamb(fa_dim, data_dim)
    q(Lamb) = N(Lamb | mean, cov)
    '''

    def __init__(self, fa_dim, data_dim, do_set_prm=False):
        '''
        mean: array(aug_dim, data_dim)
        cov: array(aug_dim, aug_dim, data_dim)
        prec: inv(cov)
        '''
        self.data_dim = data_dim
        self.fa_dim = fa_dim
        self.aug_dim = fa_dim + 1
        self.n_states = 1
        self.prior = None
        self.post = None
        if do_set_prm:
            self.set_default_params()

    def set_default_params(self):
        '''
        lamb.set_default_params()
        '''
        pm, pc = DefaultFaParams().Lamb(self.fa_dim, self.data_dim,
                                        self.n_states)
        self.set_params(mu=pm, cov=pc)

    def set_params(self, **argvs):
        self.prior = MultivariateNormal(self.data_dim, self.n_states)
        self.prior.set_params(**argvs)
        self.post = copy.deepcopy(self.prior)

    def update(self, Y, z, r):
        '''
        lamb.update(Y, z, r)
        Y: array(data_dim, data_len)
        r.post.expt: <R>
        z.post.expt2: <ZZ>
        '''
        # --- prec (inv(cov))
        rzz = einsum('ddk,ljt->ljdk', r.post.expt, z.expt2)
        prec = self.prior.prec + rzz
        # --- cov
        cov = inv(prec.transpose(2, 3, 1, 0)).transpose(2, 3, 0, 1)
        # --- mean
        yz = einsum('dt,lt->ld', Y, z.mu)
        ryz = einsum('ddk,ld->ldk', r.post.expt, yz)
        pm_ryz = self.prior.expt_prec_mu + ryz
        mean = einsum('ljdk,jdk->ldk', cov, pm_ryz)
        self.post.set_params(mu=mean, cov=cov, prec=prec)

    def samples(self, data_len=1, by_posterior=True):
        '''
        m = lamb.samples(data_len=1, by_posterior=True)
        '''
        if by_posterior:
            m = self.post.sample(data_len)
        else:
            m = self.prior.sample(data_len)
        return m

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


class qR(object):
    '''
    q(R) = Ga(R | a, b)
    <R> = a / b
    '''

    def __init__(self, data_dim, do_set_prm=False):
        '''
        a: array(data_dim, 1)
        b: array(data_dim, 1)
        mean: <R> array(data_dim) = a / b
        '''
        self.data_dim = data_dim
        self.n_states = 1
        self.prior = None
        self.post = None
        if do_set_prm:
            self.set_default_param()

    def set_default_params(self):
        a, b = DefaultFaParams.R(self.data_dim, self.n_states)
        self.set_params(a=a, b=b)

    def set_params(self, **argvs):
        self.prior = Gamma(self.n_states)
        self.prior.set_params(**argvs)
        self.post = copy.deepcopy(self.prior)
        self.data_dim = self.prior.a.shape[0]

    def update(self, Y, z, lamb):
        '''
        r.update(Y, z, lamba)
        Y: array(data_dim, data_len)
        z: <Z>, array(aug_dim, data_len)
        lamb: <lamb lamb'>, array(aug_dim, aug_dim, data_dim)
        '''
        data_len = Y.shape[-1]
        # --- a
        a = self.prior.a + 0.5 * data_len
        # --- b
        y2 = Y**2
        yz = einsum('dt,lt->dlt', Y, z.mu)
        yzl = einsum('dlt,ldk->dt', yz, lamb.post.mu)
        tr_z2l2 = einsum('ljt,jldk->dt', z.expt2, lamb.post.expt2)
        sum_y2_yzl_tr_z2l2 = 0.5 * nsum(y2 - 2 * yzl + tr_z2l2, 1)
        b = self.prior.b + sum_y2_yzl_tr_z2l2[:, newaxis]
        self.post.set_params(a=a, b=b)

    def samples(self, data_len=1, by_posterior=True):
        '''
        r = r.samples(data_len=1, by_posterior=True)
        '''
        if by_posterior:
            r = self.post.sample(data_len)
        else:
            r = self.prior.sample(data_len)
        r = einsum('dk,de->dek', r, eye(self.data_dim))
        return r

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


class Theta(object):
    '''
    theta = Theta(fa_dim, data_dim)
    '''

    def __init__(self, fa_dim, data_dim):
        '''
        theta = Theta(fa_dim, data_dim)
        '''
        self.fa_dim = fa_dim
        self.aug_dim = fa_dim + 1
        self.data_dim = data_dim
        self.lamb = qLamb(self.fa_dim, self.data_dim)
        self.r = qR(self.data_dim)
        self.update_order = ['Lamb', 'R']

    def set_default_params(self):
        '''
        theta.set_default_params()
        '''
        self.lamb.set_default_params()
        self.r.set_default_params()

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
        }
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
        self.fa_dim = self.lamb.fa_dim
        self.aug_dim = self.lamb.aug_dim
        self.data_dim = self.lamb.data_dim

    def clear_params(self):
        '''
        theta.clear_params()
        '''
        self.lamb.clear_params()
        self.r.clear_params()
        self.pi.clear_params()

    def update(self, Y, z):
        '''
        theta.update(Y, z)
        '''
        if z.mu is None:
            z.init_expt(Y.shape[-1])
        for uo in self.update_order:
            if uo == 'Lamb':
                self.lamb.update(Y, z, self.r)
            elif uo == 'R':
                self.r.update(Y, z, self.lamb)
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
            }
        '''
        dst = {
            'Lamb': self.qlamb.get_param_dict(by_posterior),
            'R': self.qr.get_param_dict(by_posterior),
        }
        return dst

    def samples(self, data_len=1, by_posterior=True):
        '''
        l, r = theta.samples(data_len=1, by_posterior=True)
        '''
        l = self.lamb.samples(data_len, by_posterior)
        r = self.r.samples(data_len, by_posterior)
        return l, r


class Fa(object):
    '''
    fa = Fa(fa_dim)
    '''

    def __init__(self, fa_dim, data_dim, **argvs):
        '''
        '''
        self.fa_dim = fa_dim
        self.aug_dim = fa_dim + 1
        self.data_dim = data_dim
        # --- theta and z
        self.z = qZ(self.fa_dim)
        self.theta = Theta(self.fa_dim, self.data_dim)
        # --- update setting
        self.max_itr = argvs.get('max_itr', 100)
        self.update_order = [
            'Z',
            'Theta',
        ]

    def init_z(self, data_len):
        self.z.init_expt(data_len)

    def set_default_params(self):
        self.theta.set_default_params()

    def set_params(self, prm):
        '''
        fa.set_params(prm)
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
        }
        '''
        self.theta.set_params(prm)

    def update(self, Y):
        '''
        fa.update(Y)
        '''
        logger.info('update order %s, in Theta %s' % (self.update_order,
                                                      self.theta.update_order))
        for i in range(self.max_itr):
            for j, uo in enumerate(self.update_order):
                if uo == 'Z':
                    self.z.update(self.theta, Y)
                elif uo == 'Theta':
                    self.theta.update(Y, self.z)
                else:
                    logger.error('%s is not supported' % uo)
                    sys.exit(-1)

    def get_param_dict(self, by_posterior=True):
        '''
        fa.get_param_dict(by_posterior=True)
        '''
        dic = self.theta.get_param_dict(by_posterior)
        return dic

    def save_param_dict(self, file_name, by_posterior):
        '''
        fa.save_param_dict(file_name)
        @argvs
        file_name: string
        '''
        dic = self.get_param_dict(by_posterior=True)
        dir_name = os.path.dirname(os.path.abspath(file_name))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(file_name, 'w') as f:
            pickle.dump(dic, f)

    def load_param_dict(self, file_name):
        '''
        fa.load_param_dict(file_name)
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

    def samples(self, data_len, by_posterior=True):
        '''
        y, z, [lamb, prec, cov] = fa.samples(data_len, by_posterior=True)

        Returns
        y: np.array(data_dim, data_len).
        z: np.array(aug_dim, data_len).
        lamb: np.array(aug_dim, data_dim, n_states).
        prec: np.array(data_dim, data_dim, n_states).
        cov: np.array(data_dim, data_dim, n_states).
        * aug_dim = fa_dim + 1
        * n_states = 1
        '''
        z = self.z.samples(data_len, by_posterior)
        lamb, prec = self.theta.samples(1, by_posterior)
        y = zeros((self.data_dim, data_len))
        mu = einsum('ld,lt->dt', lamb[:, :, 0], z)
        cov = inv(prec[:, :, 0])
        for t in range(data_len):
            y[:, t] = mvnrnd(mu[:, t], cov)
        cov = cov[:, :, newaxis]
        return y, z, [lamb, prec, cov]


def plotter(y, z, prms, title, figno=1):
    from numpy import diag
    from util.plot_models import PlotModels
    l, r, inv_r = prms
    pm = PlotModels(3, 3, figno)
    pm.plot_2d_array((0, 0), l[:-1, :, 0].T, title='$\Lambda$[:-1]')
    pm.plot_2d_array((0, 1), l[-1:, :, 0].T, title='$\Lambda$[-1]')
    pm.plot_2d_array((0, 2), diag(r[:, :, 0])[:, newaxis], title='diag(R)')
    pm.plot_2d_array((1, 0), y, title='Y')
    pm.plot_seq((1, 1), y, title='Y', cspan=2)
    pm.plot_2d_array((2, 0), z, title='Z')
    pm.plot_seq((2, 1), z, title='Z', cspan=2)
    pm.ax.legend(['%d' % x for x in range(z.shape[0])], loc=0)
    pm.sup_title(title)
    pm.tight_layout()
    pm.ion_show()


def main():
    fa_dim = 3
    data_dim = 8
    data_len = 1000
    # --- data
    fa = Fa(fa_dim, data_dim)
    fa.set_default_params()
    # fa.init_z(data_len)
    Y, Z, prms = fa.samples(data_len)
    plotter(Y, Z, prms, 'prior sample', 1)
    # --- update
    fa = Fa(fa_dim, data_dim)
    fa.set_default_params()
    # fa.init_z(data_len)
    fa.update(Y)
    Y, Z, prms = fa.samples(data_len)
    plotter(Y, Z, prms, 'posterior sample', 2)
    input('Return to finish')


if __name__ == '__main__':
    main()
