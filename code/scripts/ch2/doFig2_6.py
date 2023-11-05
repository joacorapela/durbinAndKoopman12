
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append("../../src")
import llm


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
    parser.add_argument("--forecastHorizon", type=int,
                        help="forecast horizon (samples)",
                        default=30)
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../../data/Nile.csv")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../../figures/ch2/fig2_6{:s}.{:s}")

    args = parser.parse_args()
    a1 = args.a1
    P1 = args.P1
    sigma2Epsilon = args.sigma2Epsilon
    sigma2Eta = args.sigma2Eta
    forecast_horizon = args.forecastHorizon
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    data = pd.read_csv(data_filename)
    years = data["time"].to_numpy()
    measurements = data["Nile"].to_numpy()

    measurements = np.append(measurements, np.ones(forecast_horizon)*np.nan)
    years = np.append(years, years[-1]+1+np.arange(forecast_horizon))

    if np.isnan(a1):
        a1 = measurements[0]

    # do filtering
    predicted_mean, predicted_var = a1, P1

    predicted_means = []
    predicted_vars = []
    filtered_means = []
    filtered_vars = []
    innovation_vars = []

    llmKF = llm.KalmanFilter(sigma2Epsilon=sigma2Epsilon, sigma2Eta=sigma2Eta)
    for measurement in measurements:
        predicted_means.append(predicted_mean)
        predicted_vars.append(predicted_var)
        filtered_mean, filtered_var, innovation, innovation_var, kalman_gain =\
            llmKF.update(at=predicted_mean, Pt=predicted_var, yt=measurement)
        filtered_means.append(filtered_mean)
        filtered_vars.append(filtered_var)
        innovation_vars.append(innovation_var)
        predicted_mean, predicted_var = llmKF.predict(atgt=filtered_mean,
                                                      Ptgt=filtered_var)

    # plot
    fig = go.Figure()
    trace_data = go.Scatter(x=years, y=measurements, name="measurement",
                            mode="markers", marker_color="black")
    trace_predicted_means = go.Scatter(
        x=years, y=predicted_means, name="mean prediction", mode="lines",
        line=dict(color="black", width=5.0))
    trace_predicted_means_upper = go.Scatter(
        x=years, y=predicted_means+0.675*np.sqrt(predicted_vars),
        mode="lines", line=dict(color="black", width=0.5), showlegend=False)
    trace_predicted_means_lower = go.Scatter(
        x=years, y=predicted_means-0.675*np.sqrt(predicted_vars),
        mode="lines", line=dict(color="black", width=0.5), showlegend=False)
    fig.add_trace(trace_data)
    fig.add_trace(trace_predicted_means)
    fig.add_trace(trace_predicted_means_upper)
    fig.add_trace(trace_predicted_means_lower)

    fig.update_yaxes(title="Nile River Annual Flow Volume at Aswan",
                     range=(np.nanmin(measurements), np.nanmax(measurements)))
    fig.update_xaxes(title="Year")

    png_filename = fig_filename_pattern.format("i", "png")
    html_filename = fig_filename_pattern.format("i", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    fig = go.Figure()
    trace_filterd_var = go.Scatter(x=years, y=predicted_vars,
                                   name="state variance", mode="lines",
                                   line=dict(color="black", width=1.0),
                                   showlegend=False)
    fig.add_trace(trace_filterd_var)

    fig.update_yaxes(title="Predicted State Variance", range=(5000, 50000))
    fig.update_xaxes(title="Year")

    png_filename = fig_filename_pattern.format("ii", "png")
    html_filename = fig_filename_pattern.format("ii", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
