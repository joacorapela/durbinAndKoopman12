import numpy as np
import scipy.stats


def autocorrelation(x, lags):
    xms = x - np.mean(x)
    N = len(x)
    var = np.var(x)
    ac = [1.0 if lag == 0
          else np.sum(xms[:-lag] * xms[lag:]) / N / var for lag in lags]
    return ac


def moment(x, order):
    xms = x - np.mean(x)
    moment = np.mean(xms**order)
    return moment


def skewness(ets, compute_pvalue=False):
    N = len(ets)
    m2 = moment(x=ets, order=2)
    m3 = moment(x=ets, order=3)
    S = m3 / np.sqrt(m2**3)
    if compute_pvalue:
        # S ~ N(0, 6/N)
        stat = S
        stat_mean = 0.0
        stat_var = 6.0 / N

        stat_std = np.sqrt(stat_var)
        stat_std = (stat - stat_mean) / stat_std
        pvalue = 1.0 - scipy.stats.norm.cdf(stat_std)
        return stat, pvalue
    return S


def kurtosis(ets, compute_pvalue=False):
    N = len(ets)
    m4 = moment(x=ets, order=4)
    m2 = moment(x=ets, order=2)
    stat = m4 / np.sqrt(m2**2)
    if compute_pvalue:
        # K ~ N(3, 24/N)
        stat_mean = 3.0
        stat_var = 24.0 / N

        stat_std = np.sqrt(stat_var)
        stat_std = (stat - stat_mean) / stat_std
        pvalue = 1.0 - scipy.stats.norm.cdf(stat_std)
        return stat, pvalue
    return stat


def N_statistic(ets, compute_pvalue=False):
    """
    Computes the N statistic from Durbin and Koopman 2012, p. 39
    """
    N = len(ets)
    S = skewness(ets=ets, compute_pvalue=False)
    K = kurtosis(ets=ets, compute_pvalue=False)
    stat = N * (S**2 / 6.0 + (K - 3.0)**2 / 24)
    if compute_pvalue:
        # N_statistic ~ Chi^2(2)
        df = 2

        pvalue = 1.0 - scipy.stats.chi2.cdf(stat, df)
        return stat, pvalue
    return stat


def H_statistic(ets, h, compute_pvalue=False):
    """
    Computes the H statistic from Durbin and Koopman 2012, p. 39
    """
    num = np.sum(ets[-h:]**2)
    den = np.sum(ets[:h]**2)
    stat = num / den
    if compute_pvalue:
        # H_statistic ~ F_{h,h}
        dfn = dfd = h

        pvalue = 1.0 - scipy.stats.f.cdf(stat, dfn, dfd)
        return stat, pvalue
    return stat


def Q_statistic(ets, k, compute_pvalue=False):
    """
    Computes the H statistic from Durbin and Koopman 2012, p. 39
    """
    lags = np.arange(1, k + 1)
    cs = np.array(autocorrelation(ets, lags=lags))

    N = len(ets)
    stat = N * (N + 2) * np.sum(cs**2 / np.arange(N-1, N-(k+1), -1))
    if compute_pvalue:
        # Q_statistic ~ Chi^2(k)
        df = k

        pvalue = 1.0 - scipy.stats.chi2.cdf(stat, df)
        return stat, pvalue
    return stat
