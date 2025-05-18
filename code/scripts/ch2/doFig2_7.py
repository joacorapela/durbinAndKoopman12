
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.neighbors import KernelDensity
import scipy

sys.path.append("../../src")
import llm
import time_series.utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=float, help="initial filter mean",
                        default=float("nan"))
    parser.add_argument("--P1", type=float, help="initial filter variance",
                        default=1e7)
    parser.add_argument("--sigma2Epsilon", type=float,
                        help="observations noise variance",
                        default=15099.0)
    parser.add_argument("--sigma2Eta", type=float,
                        help="state noise variance",
                        default=1469.1)
    parser.add_argument("--density_bandwidth", type=float,
                        help="bandwidth for density estimation",
                        default=.35)
    parser.add_argument("--h", type=int,
                        help="parameter for the H statistic",
                        default=33)
    parser.add_argument("--k", type=int,
                        help="parameter for the Q statistic",
                        default=9)
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../../data/Nile.csv")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../../figures/ch2/fig2_7{:s}.{:s}")
    parser.add_argument("--stats_filename", type=str,
                        help="statistic results filename",
                        default="../../../results/ch2/fig2_7_stats.txt")

    args = parser.parse_args()
    a1 = args.a1
    P1 = args.P1
    sigma2Epsilon = args.sigma2Epsilon
    sigma2Eta = args.sigma2Eta
    density_bandwidth = args.density_bandwidth
    h = args.h
    k = args.k
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern
    stats_filename = args.stats_filename

    data = pd.read_csv(data_filename)
    years = data["time"].to_numpy()
    measurements = data["Nile"].to_numpy()

    if np.isnan(a1):
        a1 = measurements[0]

    # do filtering
    llmKF = llm.KalmanFilter(sigma2Epsilon=sigma2Epsilon, sigma2Eta=sigma2Eta)
    ats, Pts = llmKF.predictBatch(y=measurements, a1=a1, P1=P1)
    vts = measurements - ats

    Fts = np.array(Pts) + sigma2Epsilon
    ets = vts / np.sqrt(Fts)

    kde = sklearn.neighbors.KernelDensity(
        kernel="gaussian", bandwidth=density_bandwidth).\
        fit(ets[:, np.newaxis])
    ets_grid = np.linspace(-3.0, 3.0, 1000)
    log_dens = kde.score_samples(ets_grid[:, np.newaxis])

    # plot
    # i
    fig = go.Figure()
    trace_std_residuals = go.Scatter(x=years, y=ets, name="residuals",
                                     mode="lines", marker_color="black")
    fig.add_trace(trace_std_residuals)
    fig.add_hline(y=0)

    fig.update_yaxes(title="Standarized residuals",
                     range=(np.nanmin(ets), np.nanmax(ets)))
    fig.update_xaxes(title="Year")

    png_filename = fig_filename_pattern.format("i", "png")
    html_filename = fig_filename_pattern.format("i", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    # ii
    fig = go.Figure()
    trace_hist = go.Histogram(x=ets, histnorm="probability density",
                              xbins=dict(
                                  start=-3.0,
                                  end=3.0,
                                  size=0.4,
                              ),
                              marker_color="white",
                              )
    fig.add_trace(trace_hist)
    fig.update_traces(marker_line_width=1, marker_line_color="black")

    trace_density = go.Scatter(x=ets_grid, y=np.exp(log_dens),
                               mode="lines", line_color="black")
    fig.add_trace(trace_density)

    fig.update(layout_showlegend=False)
    fig.update_yaxes(title="Probability")
    fig.update_xaxes(title="Innovation")

    png_filename = fig_filename_pattern.format("ii", "png")
    html_filename = fig_filename_pattern.format("ii", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    # iii
    ets_sorted = np.sort(ets)
    percents = np.arange(len(ets))/len(ets)
    std_normal_quantiles = scipy.stats.norm.ppf(percents)
    fig = go.Figure()
    trace_qq = go.Scatter(x=std_normal_quantiles, y=ets_sorted,
                          mode="lines", line_color="black")
    fig.add_trace(trace_qq)
    gridM3P3dense = np.linspace(-3, 3, 1000)
    trace_diag = go.Scatter(x=gridM3P3dense, y=gridM3P3dense, mode="lines",
                            line_color="black")
    fig.add_trace(trace_diag)

    fig.update(layout_showlegend=False)
    fig.update_xaxes(title="Theoretical Quantiles")
    fig.update_yaxes(title="Measured Quantiles")

    png_filename = fig_filename_pattern.format("iii", "png")
    html_filename = fig_filename_pattern.format("iii", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    # iv
    n_lags = 10
    lags = np.arange(1, n_lags + 1)
    ac = time_series.utils.autocorrelation(ets, lags=lags)

    fig = go.Figure()
    trace_ac = go.Bar(x=lags, y=ac, marker_color="black")
    fig.add_trace(trace_ac)

    fig.update(layout_showlegend=False)
    fig.update_xaxes(title="Lag")
    fig.update_yaxes(title="Normalized Autocorrelation", range=(-1.0, 1.0))

    png_filename = fig_filename_pattern.format("iv", "png")
    html_filename = fig_filename_pattern.format("iv", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    f = open(stats_filename, "w")

    S, S_p_value = time_series.utils.skewness(ets=ets, compute_pvalue=True)
    a_string = f"S={S}, pvalue={S_p_value}"
    f.write(a_string + '\n')
    print(a_string)

    K, K_p_value = time_series.utils.kurtosis(ets=ets, compute_pvalue=True)
    a_string = f"K={K}, pvalue={K_p_value}"
    f.write(a_string + '\n')
    print(a_string)

    N, N_p_value = time_series.utils.N_statistic(ets=ets, compute_pvalue=True)
    a_string = f"N={N}, pvalue={N_p_value}"
    f.write(a_string + '\n')
    print(a_string)

    H, H_p_value = time_series.utils.H_statistic(ets=ets, h=h,
                                                 compute_pvalue=True)
    a_string = f"H={H}, pvalue={H_p_value}"
    f.write(a_string + '\n')
    print(a_string)

    Q, Q_p_value = time_series.utils.Q_statistic(ets=ets, k=k,
                                                 compute_pvalue=True)
    a_string = f"Q={Q}, pvalue={Q_p_value}"
    f.write(a_string + '\n')
    print(a_string)

    f.close()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
