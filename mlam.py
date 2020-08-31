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






