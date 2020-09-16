#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-08-28'
__email__ = 'fredflorescfa@gmail.com'

"""Code from MLdP's Machine Learning for Asset Managers"""

import numpy as np
import pandas as pd
import re
from sklearn.neighbors import KernelDensity
import yfinance as yf
from scipy.optimize import minimize
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt


def get_daily_stock_returns(period='5y', interval='1d'):
    """ Get current S&P 500 ticker list, retrieve prices, and compute log returns.

    :param period: valid yfinance string e.g., max, 5y, 1y, ytd, etc.
    :param interval: valid yfinance string e.g., 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    :return: pandas.DataFrame, rows=daily dates, columns=stocks
    """

    # Get ticker list from Wikipedia and remove any non-standard characters
    data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    tickers = list(data[0]['Symbol'].unique())
    tickers = [x for x in tickers if re.match('^[\w-]+$', x) is not None]

    # Get daily closing prices from Yahoo! Finance in a pandas DataFrame
    prices = yf.download(tickers=tickers, period=period, interval=interval, threads=True, proxy=None)

    # Compute daily log returns
    log_returns = np.log(prices['Adj Close']).diff()

    return log_returns


def mp_pdf(var, q, pts):
    """ Marcenko-Pastur PDF
    Assume X is a TxN matrix of i.i.d. random variables
    :param var: float, variance of random variables
    :param q: T/N
    :param pts: number of samples to generate
    :return: pdf, series of float
    """
    eMin = var * (1.0 - (1.0/q) ** .5) ** 2.0
    eMax = var * (1.0 + (1.0/q) ** .5) ** 2.0

    eVal = np.linspace(eMin, eMax, pts)

    pdf = q/(2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5

    pdf = pd.Series(np.squeeze(pdf), index=np.squeeze(eVal))

    return pdf


def get_pca(matrix):
    """ Get eigenvalues and eigenvector from a Hermitian matrix.
    :param matrix: array of float
    :return:
    """
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # indices of sorted eVal in descending order
    eVal = eVal[indices]
    eVec = eVec[:, indices]
    eVal = np.diagflat(eVal)

    return eVal, eVec


def fit_kde(obs, b_width=0.25, kernel='gaussian', x=None):
    """ Fit kernel to a series of obs and derive the prob of obs
    :param obs:
    :param b_width:
    :param kernel:
    :param x:
    :return:
    """
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)

    kde = KernelDensity(kernel=kernel, bandwidth=b_width).fit(obs)

    if x is None:
        x = np.unique(obs).reshape(-1, 1)

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    log_prob = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(log_prob), index=x.flatten())

    return pdf


def get_rnd_cov(n_cols, n_facts):
    w = np.random.normal(size=(n_cols, n_facts))
    cov = np.dot(w, w.T)  # random cov mat but not full rank
    rank = np.linalg.matrix_rank(cov)
    cov += np.diag(np.random.uniform(size=n_cols))  # full rank cov
    rank_full = np.linalg.matrix_rank(cov)
    return cov


def cov2corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr <= -1] = -1
    corr[corr > 1] = 1
    return corr


def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov


def err_pdfs(var, e_val, q, b_width, pts=1000):
    """
    Fit error
    :param var:
    :param e_val:
    :param q:
    :param b_width:
    :param pts:
    :return:
    """
    pdf0 = mp_pdf(var, q, pts)  # theoretical PDF
    pdf1 = fit_kde(e_val, b_width, x=pdf0.index.values)  # empirical PDF
    sse = np.sum((pdf1-pdf0)**2)
    return sse


def find_max_eval(e_val, q, b_width):
    """
    Find max random eigenvalue by fitting Marcenko-Pastur distribution
    :param e_val:
    :param q:
    :param b_width:
    :return:
    """
    out = minimize(lambda *x: err_pdfs(*x), 0.5, args=(e_val, q, b_width),
                   bounds=((1E-5, 1-1E-5),))

    if out['success']:
        var = out['x'][0]
    else:
        var = 1

    e_max = var*(1+(1.0/q)**0.5)**2
    return e_max, var


def denoised_corr(e_val, e_vec, n_facts):
    """
    Remove noise from corr by fixing random eigenvalues.

    :param e_val:
    :param e_vec:
    :param n_facts:
    :return:
    """
    e_val_dn = np.diag(e_val).copy()
    e_val_dn[n_facts:] = e_val_dn[n_facts:].sum()/float(e_val_dn.shape[0] - n_facts)
    e_val_dn = np.diag(e_val_dn)
    corr1 = np.dot(e_vec, e_val_dn).dot(e_vec.T)
    corr1 = cov2corr(corr1)

    return corr1


def denoised_corr2(e_val, e_vec, n_facts, alpha=0):
    """
    Remove noise from corr through targeted shrinkage

    :param e_val:
    :param e_vec:
    :param n_facts:
    :param alpha:
    :return:
    """
    e_val_l = e_val[:n_facts, :n_facts]
    e_vec_l = e_vec[:, :n_facts]

    e_val_r = e_val[n_facts:, n_facts:]
    e_vec_r = e_vec[:, n_facts:]

    corr0 = np.dot(e_vec_l, e_val_l).dot(e_vec_l.T)
    corr1 = np.dot(e_vec_r, e_val_r).dot(e_vec_r.T)

    corr2 = corr0 + alpha*corr1 + np.diag(np.diag(corr1))

    return corr2


def form_block_matrix(n_blocks, b_size, b_corr):
    """
    Construct block matrices and center along diagonal of main matrix
    :param n_blocks:
    :param b_size:
    :param b_corr:
    :return:
    """

    block = np.ones((b_size, b_size)) * b_corr
    block[range(b_size), range(b_size)] = 1  # set diagonal = 1
    corr = block_diag(*([block] * n_blocks))
    return corr


def form_true_matrix(n_blocks, b_size, b_corr):
    """

    :param n_blocks:
    :param b_size:
    :param b_corr:
    :return:
    """
    corr0 = form_block_matrix(n_blocks, b_size, b_corr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.to_list()
    np.random.shuffle(cols)  # shuffles in place
    corr0 = corr0[cols].loc[cols].copy(deep=True)  # deep = True is default
    std0 = np.random.uniform(0.05, 0.2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)

    return mu0, cov0


def sim_cov_mu(mu0, cov0, n_obs, shrink=False):
    """
    
    :param mu0:
    :param cov0:
    :param n_obs:
    :param shrink:
    :return:
    """
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=n_obs)
    mu1 = x.mean(axis=0).reshape(-1, 1)
    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=0)

    return mu1, cov1


def denoise_cov(cov0, q, b_width):
    corr0 = cov2corr(cov0)
    eval0, evec0 = get_pca(corr0)
    emax0, var0 = find_max_eval(np.diag(eval0), q, b_width)
    nfacts0 = eval0.shape[0] - np.diag(eval0)[::-1].searchsorted(emax0)
    corr1 = denoised_corr(eval0, evec0, nfacts0)
    cov1 = corr2cov(corr1, np.diag(cov0) ** 0.5)

    return cov1


def opt_port(cov, mu=None):
    """
    Find portfolio weights, w, that minimizes risk for a given level of expected returns, mu.
    Procedure follows the alternative derivation in Zivot's Portfolio Matrix Theory Ch.1 (2013)

    :param cov: a square matrix, diagonal = variances and off-diagonal = pairwise covariances
    :param mu: an array of expected asset returns
    :return: w: an array of asset weights
    """
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)  # normalize weights to [0, 1]
    return w


def experiment(mu0, cov0, n_trials, shrink=False, min_var_portf=True):
    """
    Compute root mean squared errors with/without shrinkage and with/without cov denoising
    :param n_trials: integer
    :param shrink: boolean
    :param min_var_portf: boolean
    :return: rmse, rmse: float
    """

    w1 = pd.DataFrame(columns=np.arange(cov0.shape[0]),
                      index=np.arange(n_trials))
    w1_d = w1.copy(deep=True)

    # Construct
    w0 = opt_port(cov0, None if min_var_portf else mu0)
    w0 = np.repeat(w0.T, w1.shape[0], axis=0)

    # Run experiments
    n_obs = 1000
    b_width = 0.01
    np.random.seed(0)
    for i in range(n_trials):
        mu1, cov1 = sim_cov_mu(mu0, cov0, n_obs, shrink=shrink)
        if min_var_portf:
            mu1 = None
        cov1_d = denoise_cov(cov1, (n_obs * 1.0) / cov1.shape[1], b_width)

        w1.loc[i] = opt_port(cov1, mu1).flatten()
        w1_d.loc[i] = opt_port(cov1_d, mu1).flatten()

    # Compute root mean square errors
    rmse = np.mean((w1 - w0).values.flatten() ** 2) ** 0.5
    rmse_d = np.mean((w1_d - w0).values.flatten() ** 2) ** 0.5

    return rmse, rmse_d











