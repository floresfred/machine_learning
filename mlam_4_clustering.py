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
    x = ((1-corr0.fillna(0))/2.0)**0.5  # standardize corr values
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

    new_idx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[new_idx]  # reorder rows
    corr1 = corr1.iloc[:, new_idx]  # reorder columns

    clstrs = {i : corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist()
              for i in np.unique(kmeans.labels_)}

    silh = pd.Series(silh, index=x.index)

    return corr1, clstrs, silh
