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


def cluster_kmeans_base(corr0, max_num_clusters=10, n_init=10):
    """
    An application of sklearn.cluster.KMeans that
        1] searches for the optimal number of clusters based on the Silhouette score
        2] mitigates the randomness of the centroid initialization

    :param corr0: array of float (NxN), a square, nondegenerate correlation matrix (rank>2)
    :param max_num_clusters: int, the upper limit on the optimal cluster number search
    :param n_init: int, the number of centroid seed initializations
    :return: corr1: array of float (NxN), a copy of corr0 that is reordered by cluster.
             clstrs, dict,
                keys: int, cluster label
                values: list of int, the index members of each cluster
                silh: array of float, the Silhouette Score by index
    """

    # Transform the linear correlation matrix to a metric-based codependence matrix.
    # (see ch.3)
    x = ((1-corr0.fillna(0))/2.0)**0.5

    silh = pd.Series()

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
    if max_num_clusters is None:
        max_num_clusters = corr0.shape[1] - 1
    corr1, clstrs, silh = cluster_kmeans_base(corr0,
                                              max_num_clusters=min(max_num_clusters, corr0.shape[1]-1),
                                              n_init=n_init)
    cluster_tstats = {i: np.mean(silh[clstrs[i]])/np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tstat_mean = sum(cluster_tstats.values())/len(cluster_tstats)
    redo_clusters = [i for i in cluster_tstats.keys() if cluster_tstats[i]<tstat_mean]
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
