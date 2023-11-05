
import sys
import argparse
import pickle
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
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../../data/Nile.csv")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../../figures/ch2/fig2_1{:s}.{:s}")
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../../results/Nile_a1{:.02f}_P1{:02f}_s2Epsilon{:02f}_s2Eta{:02f}.pickle")

    args = parser.parse_args()
    a1 = args.a1
    P1 = args.P1
    sigma2Epsilon = args.sigma2Epsilon
    sigma2Eta = args.sigma2Eta
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern
    results_filename = args.results_filename_pattern.format(a1, P1,
                                                            sigma2Epsilon,
                                                            sigma2Eta)
    data = pd.read_csv(data_filename)
    years = data["time"]
    measurements = data["Nile"]

    if np.isnan(a1):
        a1 = measurements[0]

    # do filtering
    predicted_mean, predicted_var  = a1, P1

    predicted_means = []
    predicted_vars = []
    filtered_means = []
    filtered_vars = []
    innovations = []
    innovation_variances = []
    kalman_gains = []

    llmKF = llm.KalmanFilter(sigma2Epsilon=sigma2Epsilon, sigma2Eta=sigma2Eta)
    for measurement in measurements:
        predicted_means.append(predicted_mean)
        predicted_vars.append(predicted_var)
        filtered_mean, filtered_var, innovation, innovation_var, kalman_gain = \
            llmKF.update(at=predicted_mean, Pt=predicted_var, yt=measurement)
        filtered_means.append(filtered_mean)
        filtered_vars.append(filtered_var)
        innovations.append(innovation)
        innovation_variances.append(innovation_var)
        kalman_gains.append(kalman_gain)
        predicted_mean, predicted_var = llmKF.predict(atgt=filtered_mean,
                                                      Ptgt=filtered_var)

    # save results
    with open(results_filename, "wb") as f:
        pickle.dump({"sigma2Epsilon": sigma2Epsilon, "sigma2Eta": sigma2Eta,
                     "at": predicted_means, "atgt": filtered_means,
                     "Pt": predicted_vars, "Ptgt": filtered_vars,
                     "vt": innovations, "Ft": innovation_variances,
                     "Kt": kalman_gains}, f)

    # plot
    fig = go.Figure()
    trace_data = go.Scatter(x=years, y=measurements, name="measurement",
                            mode="markers", marker_color="black")
    trace_predicted_means = go.Scatter(x=years, y=predicted_means,
                                      name="mean prediction", mode="lines",
                                      line=dict(color="black", width=5.0))
    trace_predicted_means_upper = go.Scatter(x=years,
                                            y=predicted_means+1.65*np.sqrt(predicted_vars),
                                            mode="lines",
                                            line=dict(color="black",
                                                      width=0.5),
                                            showlegend=False)
    trace_predicted_means_lower = go.Scatter(x=years,
                                            y=predicted_means-1.65*np.sqrt(predicted_vars),
                                            mode="lines",
                                            line=dict(color="black",
                                                      width=0.5),
                                            showlegend=False)
    fig.add_trace(trace_data)
    fig.add_trace(trace_predicted_means)
    fig.add_trace(trace_predicted_means_upper)
    fig.add_trace(trace_predicted_means_lower)

    fig.update_yaxes(title="Nile River Annual Flow Volume at Aswan", range=(np.min(measurements), np.max(measurements)))
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

    fig.update_yaxes(title="Predicted State Variance", range=(5000, 17500))
    fig.update_xaxes(title="Year")

    png_filename = fig_filename_pattern.format("ii", "png")
    html_filename = fig_filename_pattern.format("ii", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    fig = go.Figure()
    trace_innovations = go.Scatter(x=years, y=innovations,
                                   name="innovations", mode="lines",
                                   line=dict(color="black", width=1.0))
    fig.add_trace(trace_innovations)
    fig.add_hline(y=0)

    fig.update_yaxes(title="Innovation")
    fig.update_xaxes(title="Year")

    png_filename = fig_filename_pattern.format("iii", "png")
    html_filename = fig_filename_pattern.format("iii", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    fig = go.Figure()
    trace_innovations = go.Scatter(x=years, y=innovation_variances,
                                   mode="lines",
                                   line=dict(color="black", width=1.0),
                                   showlegend=False)
    fig.add_trace(trace_innovations)

    fig.update_yaxes(title="Innovation Variance", range=(20000, 32500))
    fig.update_xaxes(title="Year")

    png_filename = fig_filename_pattern.format("iv", "png")
    html_filename = fig_filename_pattern.format("iv", "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)
    fig.show()

    breakpoint()

if __name__ == "__main__":
   main(sys.argv)
