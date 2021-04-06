#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
kl_divergense.py
'''
import os
import sys
from numpy import exp
from numpy import log
from numpy import trace
from numpy import dot
from numpy import einsum
from scipy.special import digamma
from scipy.special import gammaln

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))
from utils.calc_utils import inv
from utils.calc_utils import logdet


def KL_Dir(ln_alpha_h, ln_alpha):
    '''
    :param ln_alpha_h: arr(n_states),
    :param ln_alpha: arr(n_states),
    '''
    n_states = len(ln_alpha)
    alpha_h = exp(ln_alpha_h)
    alpha = exp(ln_alpha)
    alpha_0_h = sum(alpha_h)
    alpha_0 = sum(alpha)
    dig_alpha_0_h = digamma(alpha_0_h)
    val = gammaln(alpha_0_h) - gammaln(alpha_0)
    for k in range(n_states):
        val += gammaln(alpha[k]) - gammaln(alpha_h[k])
        val += (alpha_h[k] - alpha[k]) * (digamma(alpha_h[k]) - dig_alpha_0_h)
    return val


def KL_Gamma(ln_a_h, ln_b_h, ln_a, ln_b):
    '''
    :param ln_a_h: float
    :param ln_b_h: float
    :param ln_a: float
    :param ln_b: float
    :return val: float
    '''
    a_h = exp(ln_a_h)
    a = exp(ln_a)
    val = (-gammaln(a_h) + gammaln(a) + a * (ln_b_h - ln_b) +
           (a_h - a) * digamma(a_h) - a_h + exp(ln_b + ln_a_h - ln_b_h))
    return val


def KL_Gauss(mu_h, sig_h, mu, sig):
    '''
    :param mu_h: arr(data_dim)
    :param sig_h: arr(data_dim, data_dim)
    :param mu: arr(data_dim)
    :param sig: arr(data_dim, data_dim)
    :return val: float
    (mu_h - mu)T * sig^{-1} * (mu_h - mu)
    + Tr(inv_sig * sig_h) - data_dim + log(|sig|) - log(|sig_h|)
    '''
    data_dim = len(mu_h)
    inv_sig = inv(sig)
    mu_h_mu = mu_h - mu
    val = 0.5 * (einsum('d,de,e->', mu_h_mu, inv_sig, mu_h_mu) + trace(
        dot(inv_sig, sig_h)) - data_dim + logdet(sig) - logdet(sig_h))
    return val


def KL_Norm_Wish(beta_h, mu_h, nu_h, W_h, beta, mu, nu, W):
    '''
    :param beta_h/ beta: float
    :param mu_h/ m: arr(D, )
    :param nu_h/ nu: float
    :param W_h/ W: arr(D, D)
    :return val: float
    '''
    data_dim = len(mu)
    val = 0
    if mu.ndim == 1:
        R = nu_h * W_h
        m_diff = mu_h - mu
        m_diff2 = einsum('i,j->ij', m_diff, m_diff)
        m_diff2_R = einsum('ij,ji->...', m_diff2, beta * R)
        val += 0.5 * (m_diff2_R + data_dim * log(beta_h / beta) + data_dim *
                      (beta / beta_h - 1))
    else:
        m_diff = mu_h - mu
        m_diff2 = einsum('dk,ek->dek', m_diff, m_diff)
        R = einsum('k,dek->dek', nu_h, W_h)
        b_R = einsum('k,dek->dek', beta, R)
        m_diff2_R = einsum('dek,edk->...', m_diff2, b_R)
        d_ln_bh_b = data_dim * log(beta_h / beta)
        d_b_bh = data_dim * beta / beta_h
        val += 0.5 * (m_diff2_R + d_ln_bh_b + d_b_bh)
    val += KL_Wishart(nu_h, W_h, nu, W)
    return val


def KL_Wishart(nu_h, W_h, nu, W):
    '''
    :param nu_h/ nu: float
    :param W_h/ W: arr(D, D)
    :return val: float
    '''
    from numpy import newaxis
    data_dim = len(W[0])
    val = 0
    if W.ndim == 2:
        invW = inv(W)
        invW_wh = dot(invW, W_h)
        val += -0.5 * nu * logdet(invW_wh[newaxis, :, :])
        val += 0.5 * nu_h * (trace(invW_wh) - data_dim)
        for d in range(1, data_dim + 1):
            val += gammaln((nu + 1 - d) / 2.0) - gammaln((nu_h + 1 - d) / 2.0)
            val += (nu_h - nu) * digamma((nu_h + 1.0 - d) / 2.0) / 2.0
    else:
        invW = inv(W.transpose(2, 0, 1)).transpose(1, 2, 0)
        invW_Wh = einsum('dek,efk->kdf', invW, W_h)
        val += -0.5 * nu * logdet(invW_Wh)
        val += 0.5 * nu_h * (trace(invW_Wh) - data_dim)
        nuh_nu = nu_h - nu
        for d in range(1, data_dim + 1):
            lngam_nuh_d_2 = gammaln((nu_h + 1.0 - d) / 2.0)
            lngam_nu_d_2 = gammaln(nu + 1.0 - d / 2.0)
            dig_nuh_d_2 = digamma((nu_h + 1.0 - d) / 2.0)
            val += lngam_nu_d_2 - lngam_nuh_d_2
            val += nuh_nu * dig_nuh_d_2
    return val
