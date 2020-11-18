#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-11-17'
__email__ = 'fredflorescfa@gmail.com'

"""Code from MLdP's Ch.3 Machine Learning for Asset Managers"""

import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
from matplotlib_venn import venn2


def kl_divergence(px, qx):
    """ The Kullback-Leibler (KL) Divergence. The difference between two discrete probability
        distributions, px and qx.

    :param px: numpy array of float, probability distribution, np.sum(px) = 1.0
    :param qx: numpy array of float, probability distribution, np.sum(qx) = 1.0
    :return: float
    """
    px = px/np.sum(px)
    qx = qx/np.sum(qx)
    return -np.sum(px*np.log(qx/px))


def marginal_entropy(px):
    """ The amount of uncertainty associated with a discrete random variable x.
        It is the expected value of all surprising observations of x.

    :param px: numpy array of float, probability distribution, np.sum(px) = 1.0
    :return:
    """
    # identical to scipy.stats.entropy(px)
    px = px/np.sum(px)
    return -np.sum(px*np.log(px))


def cross_entropy(px, qx):
    px = px/np.sum(px)
    qx = qx/np.sum(qx)
    return -np.sum(px*np.log(qx))


def num_bins(n_obs, corr=None):
    # Compute optimal number of bins for discretization
    if corr is None:
        z = (8+324*n_obs+12*(36*n_obs+729*n_obs**2)**0.5)**(1/3.)
        b = round(z/6.+2./(3*z)+1./3)
    else:
        b = round((1/(2**.5))*(1+(1+24*n_obs/(1.-corr**2))**.5)**.5)
    return int(b)


def mutual_information(x, y, norm=False):
    bXY = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cXY = np.histogram2d(x, y, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    if norm:
        hX = ss.entropy(np.histogram(x, bXY)[0])
        hY = ss.entropy(np.histogram(y, bXY)[0])
        iXY /= min(hX, hY)
    return iXY


def variation_of_information(x, y, norm=False):
    bXY = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cXY = np.histogram2d(x, y, bXY)[0]
    hX = ss.entropy(np.sum(cXY, axis=1))
    hY = ss.entropy(np.sum(cXY, axis=0))
    iXY = mutual_info_score(labels_true=None, labels_pred=None, contingency=cXY)
    vXY = hX + hY - 2 * iXY
    if norm:
        hXY = hX + hY - iXY
        vXY /= hXY
    return vXY


def exercise_1(size):
    # Measure entropies between two normal random variables

    rows = ['N', 'BIN[X] = BIN[Y]', 'BIN[X,Y]', 'CORR[X,Y]', 'H[X]', 'H[Y]', 'H[X,Y]',
            'H[X|Y]', 'I[X,Y]', 'VI[X,Y]', 'VI_N[X,Y]']
    cols = ['-1.0', '-0.5', '0', '+0.5', '+1.0']
    col_idx = [-50.0, -0.6, 0.0, 0.6, 50.0]  # multipliers that targets each correlation
    df = pd.DataFrame(index=rows, columns=cols)

    seed = 94563
    np.random.seed(seed)

    x = np.random.normal(loc=0.0, scale=1.0, size=size)
    e = np.random.normal(loc=0.0, scale=1.0, size=size)
    binX = num_bins(size, corr=None)  # = binY, the optimal number of bins for marginal cases

    for c, m in zip(cols, col_idx):
        y = m * x + e

        rho = np.corrcoef(x, y)[0, 1]
        binXY = num_bins(size, corr=rho)  # the optimal number of bins for joint case

        df.loc['N', c] = size
        df.loc['BIN[X] = BIN[Y]', c] = binX
        df.loc['BIN[X,Y]', c] = binXY
        df.loc['CORR[X,Y]', c] = rho

        df.loc['H[X]', c] = ss.entropy(np.histogram(x, binX)[0])  # will be constant
        df.loc['H[Y]', c] = ss.entropy(np.histogram(y, binX)[0])  # will vary with correlation

        df.loc['I[X,Y]', c] = mutual_information(x, y, norm=True)

    df.loc['H[X,Y]', :] = df.loc['H[X]'] + df.loc['H[Y]'] - df.loc['I[X,Y]']
    df.loc['H[X|Y]', :] = df.loc['H[X,Y]'] - df.loc['H[Y]']
    df.loc['VI[X,Y]', :] = df.loc['H[X]'] + df.loc['H[Y]'] - 2 * df.loc['I[X,Y]']
    df.loc['VI_N[X,Y]', :] = df.loc['VI[X,Y]'] / df.loc['H[X,Y]']

    return df


def venn_entropy(num_obs):
    # Plot VENN diagram
    df = exercise_1(num_obs)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for p, j in zip(['+1.0', '+0.5', '0'], [0, 1, 2]):
        rho = round(df.loc['CORR[X,Y]', p], 4)
        hX = round(df.loc['H[X]', p], 4)
        hY = round(df.loc['H[Y]', p], 4)
        iXY = round(df.loc['I[X,Y]', p], 4)
        v = venn2(subsets=(hX, hY, iXY), set_labels=(r'$H[X]$', r'$H[Y]$'), ax=axes[j])
        # v.get_label_by_id('11').set_text(r'$I[X,Y]$ = {:.4f}'.format(iXY))
        axes[j].set_title(r'$\rho$ = {:.4f}'.format(rho), fontsize=14)
    axes[0].annotate(r'$I[X,Y]$', xy=v.get_label_by_id('11').get_position(), xytext=(0, -25),
                     ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0))
    plt.suptitle(f'Entropies for {num_obs:,} Observations')
    plt.show()

