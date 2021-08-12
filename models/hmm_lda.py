#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
cluster_lda.py
'''
import os
import sys

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cdir)
from hmm import Hmm
from gmm import Gmm
from lda import Lda

sys.path.append(os.path.join(cdir, '..'))
from python_utilities.utils.logger import logger


def _data_len_ibgn_iend_itr(data_lens):
    ibgn = 0
    for n in data_lens:
        iend = ibgn + n
        yield ibgn, iend
        ibgn = iend


def _mu_by_kmeans(data, labels, n_states):
    from sklearn.cluster import KMeans
    km = KMeans(n_states)
    km.fit(data)
    mu = km.cluster_centers_.T
    return mu


def _prior_mu(data, labels, n_states):
    data_len, data_dim = data.shape
    # mu = randn(data_dim, n_states) * 2
    # mu = np.zeros((data_dim, n_states))
    # mu = randn(data_dim, n_states)
    # mu = _mu_by_gmm_each_lbl(data, labels, n_states)
    mu = _mu_by_kmeans(data, labels, n_states)
    return mu


class ClusterLda():
    def __init__(self, **args):
        self._cluster_dic = {}
        self._lda_dic = {}
        self._cluster_em_order = args.get('cluster_em_order', 'ME')
        # self._cluster_mdl_name = args.get('cluster_mdl_name', 'hmm')
        self._cluster_mdl_name = args.get('cluster_mdl_name', 'gmm')

        if self._cluster_mdl_name == 'hmm':
            self._cluster_mdl = Hmm
        if self._cluster_mdl_name == 'gmm':
            self._cluster_mdl = Gmm
        else:
            raise Exception('Not Supported: %s' % self._cluster_mdl_name)

    def update(self,
               data,
               data_lens,
               n_states,
               n_cat,
               n_cluster_em_itr,
               n_lda_em_itr,
               labels=None):
        cluster = self.update_cluster(data, n_states, n_cluster_em_itr, labels)
        lda = self.update_lda(cluster, data, data_lens, n_cat, n_lda_em_itr, labels)
        return cluster, lda

    def n_cat_list(self):
        n_cat_list = [k[1] for k in self._lda_dic.keys()]
        return n_cat_list

    def estimate(self, data, n_states, n_cat):
        '''
        estimate category of 1 data chunk
        '''
        cluster = self._cluster_dic.get(n_states, None)
        lda = self._lda_dic.get((n_states, n_cat), None)
        if cluster is None:
            err_msg = 'Cluster model %d states has not been trained' % n_states
            logger.error(err_msg)
            return None
        if lda is None:
            err_msg = 'LDA (%d, %d) states has not been trained' % (n_states,
                                                                    n_cat)
            logger.error(err_msg)
            return None
        else:
            est_x, est_s, vb = cluster.estimate(data.T)
            cluster_prms = cluster.get_params()
            s_list, z_list, lda_prms = lda.predict([est_s])
            states = s_list[0]
            category = z_list[0]
        return est_x, states, category, cluster_prms, lda_prms

    def cluster_mdl(self, n_states):
        cluster = self._cluster_dic.get(n_states, None)
        if cluster is None:
            err_msg = 'cluster model %d states has not been trained' % n_states
            logger.error(err_msg)
            return None
        return cluster

    def lda_mdl(self, n_states, n_cat):
        lda = self._lda_dic.get((n_states, n_cat), None)
        if lda is None:
            err_msg = 'LDA (%d, %d) states has not been trained' % (n_states,
                                                                    n_cat)
            logger.error(err_msg)
            return None
        return lda

    def update_cluster(self, data, n_states, n_em_itr, labels=None):
        cluster = self.cluster_mdl(n_states)
        if cluster is None:
            data_len, data_dim = data.shape
            if self._cluster_em_order == 'EM':
                uo = ['E', 'M']
                mu = _prior_mu(data, labels, n_states)
                prms = {'expt_init_mode': 'random', 'update_order': uo}
                mdl = self._cluster_mdl(data_dim, n_states, **prms)
                mdl.set_params({'MuR': {'mu': mu}})
            elif self._cluster_em_order == 'ME':
                uo = ['M', 'E']
                prms = {'expt_init_mode': 'kmeans', 'update_order': uo}
                mdl = self._cluster_mdl(data_dim, n_states, **prms)
                mdl.set_params({})
                mdl.init_expt_s(data_len, data.T)
        # --- update
        mdl.update(data.T, n_em_itr)
        self._cluster_dic[n_states] = mdl
        return mdl

    def update_lda(self,
                   cluster,
                   data,
                   data_lens,
                   n_cat,
                   n_em_itr,
                   labels=None):
        n_states = cluster.n_states
        lda = self.lda_mdl(n_states, n_cat)
        if lda is None:
            states = cluster.expt_s.argmax(0)
            s_list = [
                states[b:e] for b, e in _data_len_ibgn_iend_itr(data_lens)
            ]
            lda = Lda(n_states, n_cat)
            n_batches = len(data_lens)
            lda.set_default_params(n_batches)
            lda.init_expt_z(data_lens)
        lda.update(s_list, n_em_itr)
        self._lda_dic[(n_states, n_cat)] = lda
        return lda


def main():
    from numpy.random import randn
    data_lens = [200, 300]
    data_len = sum(data_lens)
    data_dim = 8
    n_states = 4
    n_cat = 3
    data = randn(data_len, data_dim)
    cluster_lda = ClusterLda()
    cluster_lda.update_cluster(data, n_states)
    cluster = cluster_lda.cluster_mdl(n_states)
    cluster_lda.update_lda(cluster, data, data_lens, n_cat)
    ret = cluster_lda.estimate(data, n_states, n_cat)
    print(ret)


if __name__ == '__main__':
    main()
