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

    pdf = pd.Series(pdf, index=eVal)

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
