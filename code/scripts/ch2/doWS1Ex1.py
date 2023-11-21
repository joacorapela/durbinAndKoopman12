
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
    parser.add_argument("--vars_thr", type=float,
                        help=("threshold for similarity between KF and "
                              "steady-state variance"),
                        default=1e-3)
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../../data/Nile.csv")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../../figures/ch2/figWS1_ex1{:s}.{:s}")

    args = parser.parse_args()
    a1 = args.a1
    P1 = args.P1
    sigma2Epsilon = args.sigma2Epsilon
    sigma2Eta = args.sigma2Eta
    vars_thr = args.vars_thr
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    data = pd.read_csv(data_filename)
    years = data["time"]
    measurements = data["Nile"]

    if np.isnan(a1):
        a1 = measurements[0]

    # do filtering
    llmKF = llm.KalmanFilter(sigma2Epsilon=sigma2Epsilon, sigma2Eta=sigma2Eta)
    predicted_means, predicted_vars = llmKF.predictBatch(y=measurements,
                                                         a1=a1, P1=P1)
    q = sigma2Eta / sigma2Epsilon
    x = (q + np.sqrt(q**2 + 4*q)) / 2.0
    Pss = x * sigma2Epsilon
    vars_sim_index = np.argmax((predicted_vars - Pss) < vars_thr)

    ss_predicted_means, ss_predicted_vars = llmKF.steadyStatePredictBatch(
        y=measurements, a1=a1, P1=P1)

    mse = np.mean((np.array(predicted_means)-np.array(ss_predicted_means))**2)

    # plot (a)
    fig = go.Figure()
    trace_filterd_var = go.Scatter(x=years, y=predicted_vars,
                                   name="state variance", mode="lines",
                                   line=dict(color="black", width=1.0),
                                   showlegend=False)
    fig.add_trace(trace_filterd_var)
    fig.add_hline(y=Pss, line_dash="dash", line_color="gray")
    fig.add_vline(x=years[vars_sim_index], line_dash="dash", line_color="gray")

    fig.update_yaxes(title="Predicted State Variance", range=(5000, 17500))
    fig.update_xaxes(title="Year")

    png_filename = fig_filename_pattern.format("a", "png")
    html_filename = fig_filename_pattern.format("a", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    # plot (b)
    fig = go.Figure()
    trace_data = go.Scatter(x=years, y=measurements, name="measurement",
                            mode="markers", marker_color="black")
    trace_predicted_means = go.Scatter(x=years, y=predicted_means,
                                       name="filter", mode="lines",
                                       legendgroup="filter",
                                       line=dict(color="black", width=5.0))
    trace_predicted_means_upper = go.Scatter(x=years,
                                             y=(predicted_means +
                                                1.65*np.sqrt(predicted_vars)),
                                             mode="lines",
                                             line=dict(color="black",
                                                       width=0.5),
                                             legendgroup="filter",
                                             showlegend=False)
    trace_predicted_means_lower = go.Scatter(x=years,
                                             y=(predicted_means -
                                                1.65*np.sqrt(predicted_vars)),
                                             mode="lines",
                                             line=dict(color="black",
                                                       width=0.5),
                                             legendgroup="filter",
                                             showlegend=False)
    trace_ss_predicted_means = go.Scatter(x=years, y=ss_predicted_means,
                                          name="ss filter", mode="lines",
                                          legendgroup="ss_filter",
                                          line=dict(color="blue", width=5.0))
    trace_ss_predicted_means_upper = go.Scatter(x=years,
                                                y=(ss_predicted_means +
                                                   1.65*np.sqrt(ss_predicted_vars)),
                                                mode="lines",
                                                line=dict(color="blue",
                                                          width=0.5),
                                                legendgroup="ss_filter",
                                                showlegend=False)
    trace_ss_predicted_means_lower = go.Scatter(x=years,
                                                y=(ss_predicted_means -
                                                   1.65*np.sqrt(ss_predicted_vars)),
                                                mode="lines",
                                                line=dict(color="blue",
                                                          width=0.5),
                                                legendgroup="ss_filter",
                                                showlegend=False)
    fig.add_trace(trace_data)
    fig.add_trace(trace_predicted_means)
    fig.add_trace(trace_predicted_means_upper)
    fig.add_trace(trace_predicted_means_lower)
    fig.add_trace(trace_ss_predicted_means)
    fig.add_trace(trace_ss_predicted_means_upper)
    fig.add_trace(trace_ss_predicted_means_lower)

    fig.update_layout(title=f"MSE={mse}")
    fig.update_yaxes(title="Predicted State Variance",
                     range=(min(measurements), max(measurements)))
    fig.update_xaxes(title="Year")

    png_filename = fig_filename_pattern.format("b", "png")
    html_filename = fig_filename_pattern.format("b", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
