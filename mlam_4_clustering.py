#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-11-18'
__email__ = 'fredflorescfa@gmail.com'

"""Code from MLdP's Ch.4 Machine Learning for Asset Managers"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.utils import check_random_state
from scipy.linalg import block_diag
from machine_learning.mlam_2_denoise_detone import cov2corr


def cluster_kmeans_base(corr0, max_num_clusters=10, n_init=10):
    """
    An application of sklearn.cluster.KMeans that
        1] searches for the optimal number of clusters based on the Silhouette score
        2] mitigates the randomness of KMeans' initialization by running alternative seeds.

    :param corr0: array of float (NxN), a square, non-degenerate correlation matrix (rank>2)
    :param max_num_clusters: int, the upper limit on the optimal cluster number search
    :param n_init: int, the number of centroid seed initializations
    :return: corr1: array of float (NxN), a copy of corr0 that is reordered by cluster.
             clstrs, dict,
                keys: int, cluster label
                values: list of int, the index members of each cluster
            silh: array of float, the Silhouette Score by index member
    """

    # Transform the linear correlation matrix to a metric-based codependence matrix.
    # (see ch.3)
    x = ((1-corr0.fillna(0))/2.0)**0.5

    silh = pd.Series()
    kmeans = None

    for init in range(n_init):
        for i in range(2, max_num_clusters+1):
            kmeans_ = KMeans(n_clusters=i, n_jobs=1, n_init=1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())
            if np.isnan(stat[1]) or (stat[0] > stat[1]):
                silh = silh_
                kmeans = kmeans_

    # Reorganize correlation matrix by cluster.
    new_idx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[new_idx]  # reorder rows
    corr1 = corr1.iloc[:, new_idx]  # reorder columns

    # Create dictionary of clusters (key) and index membership (value)
    clstrs = {i : corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist()
              for i in np.unique(kmeans.labels_)}

    silh = pd.Series(silh, index=x.index)

    return corr1, clstrs, silh


def make_new_outputs(corr0, clstrs, clstrs2):
    clstrs_new = {}
    for i in clstrs.keys():
        clstrs_new[len(clstrs_new.keys())] = list(clstrs[i])
    for i in clstrs2.keys():
        clstrs_new[len(clstrs_new.keys())] = list(clstrs2[i])
    new_idx = [j for i in clstrs_new for j in clstrs_new[i]]
    corr_new = corr0.loc[new_idx, new_idx]
    x = ((1-corr0.fillna(0))/2.0)**0.5
    kmeans_labels = np.zeros(len(x.columns))
    for i in clstrs_new.keys():
        idxs = [x.index.get_loc(k) for k in clstrs_new[i]]
        kmeans_labels[idxs] = i
    silh_new = pd.Series(silhouette_samples(x, kmeans_labels), index=x.index)
    return corr_new, clstrs_new, silh_new


def cluster_kmeans_top(corr0, max_num_clusters=None, n_init=10):
    """
    A recursive function to re-cluster the observations of all original clusters whose average Silhouette t-stat
    falls below the average cluster.

    :param corr0: array of float (NxN), a square, non-degenerate correlation matrix (rank>2)
    :param max_num_clusters: int, the upper limit on the optimal cluster number search
    :param n_init: int, the number of centroid seed initializations
    :return: corr_new: array of float (NxN), a copy of corr0 that is reordered by cluster.
             clstrs_new, dict,
                keys: int, cluster label
                values: list of int, the index members of each cluster
            silh_new: array of float, the Silhouette Score by index member
    """
    if max_num_clusters is None:
        max_num_clusters = corr0.shape[1] - 1

    corr1, clstrs, silh = cluster_kmeans_base(corr0,
                                              max_num_clusters=min(max_num_clusters, corr0.shape[1]-1),
                                              n_init=n_init)

    cluster_tstats = {i: np.mean(silh[clstrs[i]])/np.std(silh[clstrs[i]])
        if np.isnan(np.std(silh[clstrs[i]])) else np.nan for i in clstrs.keys()}
    tstat_mean = sum(cluster_tstats.values())/len(cluster_tstats)
    redo_clusters = [i for i in cluster_tstats.keys() if cluster_tstats[i] < tstat_mean]
    if len(redo_clusters) <= 1:
        return corr1, clstrs, silh
    else:
        keys_redo = [j for i in redo_clusters for j in clstrs[i]]
        corr_temp = corr0.loc[keys_redo, keys_redo]
        tstat_mean = np.mean([cluster_tstats[i] for i in redo_clusters])
        corr2, clstrs2, silh2 = cluster_kmeans_top(corr_temp,
                                                   max_num_clusters=min(max_num_clusters, corr_temp.shape[1]-1),
                                                   n_init=n_init)
        corr_new, clstrs_new, silh_new = make_new_outputs(corr0,
                                                          {i: clstrs[i] for i in clstrs.keys()
                                                           if i not in redo_clusters},
                                                          clstrs2)
        tstat_mean_new = np.mean([np.mean(silh_new[clstrs_new[i]]) /
                                  np.std(silh_new[clstrs_new[i]]) for i in clstrs_new.keys()])
        if tstat_mean_new <= tstat_mean:
            return corr1, clstrs, silh
        else:
            return corr_new, clstrs_new, silh_new


def get_cov_sub(n_obs, n_cols, sigma, random_state=None):

    rng = check_random_state(random_state)
    if n_cols == 1:
        return np.ones((1, 1))
    ar0 = rng.normal(size=(n_obs, 1))
    ar0 = np.repeat(ar0, n_cols, axis=1)
    ar0 += rng.normal(scale=sigma, size=ar0.shape)
    ar0 = np.cov(ar0, rowvar=False)
    return ar0


def get_rnd_block_cov(n_cols, n_blocks, min_block_size=1, sigma=1.0, random_state=None):

    rng = check_random_state(random_state)
    parts = rng.choice(range(1, n_cols-(min_block_size-1)*n_blocks), n_blocks-1, replace=False)
    parts.sort()
    parts = np.append(parts, n_cols-(min_block_size-1)*n_blocks)
    parts = np.append(parts[0], np.diff(parts)) - 1 + min_block_size
    cov = None
    for n_cols_ in parts:
        cov_ = get_cov_sub(int(max(n_cols * (n_cols_ + 1)/2.0, 100)), n_cols_, sigma, random_state=rng)
        if cov is None:
            cov = cov_.copy()
        else:
            cov = block_diag(cov, cov_)
    return cov


def random_block_corr(n_cols, n_blocks, random_state=None, min_block_size=1):

    rng = check_random_state(random_state)
    cov0 = get_rnd_block_cov(n_cols, n_blocks, min_block_size=min_block_size, sigma=0.5, random_state=rng)
    cov1 = get_rnd_block_cov(n_cols, 1, min_block_size=min_block_size, sigma=1.0, random_state=rng)
    cov0 += cov1
    corr0 = cov2corr(cov0)
    corr0 = pd.DataFrame(corr0)
    return corr0
