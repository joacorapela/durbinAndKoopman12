
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append("../src")
import tsa

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../data/Nile.csv")
    parser.add_argument("--filtering_results_filename", type=str,
                        help="filtering results filename",
                        default="../../results/Nile_a1nan_P110000000.000000_s2Epsilon15099.000000_s2Eta1469.100000.pickle")
    parser.add_argument("--smoothing_results_filename", type=str,
                        help="smoothing results filename",
                        default="../../results/Nile_a1nan_P110000000.000000_s2Epsilon15099.000000_s2Eta1469.100000_smoothing.pickle")

    args = parser.parse_args()
    data_filename = args.data_filename
    filtering_results_filename = args.filtering_results_filename
    smoothing_results_filename = args.smoothing_results_filename

    data = pd.read_csv(data_filename)
    years = data["time"]
    measurements = data["Nile"]

    with open(filtering_results_filename, "rb") as f:
        filtering_results = pickle.load(f)
    sigma2Epsilon = filtering_results["sigma2Epsilon"]
    sigma2Eta = filtering_results["sigma2Eta"]
    at = filtering_results["at"]
    Pt = filtering_results["Pt"]
    atgt = filtering_results["atgt"]
    Ptgt = filtering_results["Ptgt"]

    llmKF = tsa.LocalLevelModelKalmanFilter(sigma2Epsilon=sigma2Epsilon,
                                            sigma2Eta=sigma2Eta)
    alphaHat, Vt = llmKF.smooth(ys=measurements, at=at, Pt=Pt, atgt=atgt,
                                Ptgt=Ptgt)
    smoothed_means = alphaHat
    smoothed_vars = Vt

    # save results
    with open(smoothing_results_filename, "wb") as f:
        pickle.dump({"filtering_results_filename": filtering_results_filename,
                     "alphaHat": alphaHat, "Vt": Vt}, f)

    # plot
    fig = go.Figure()
    trace_data = go.Scatter(x=years, y=measurements, name="measurement",
                            mode="markers", marker_color="black")
    trace_smoothed_means = go.Scatter(x=years, y=smoothed_means,
                                      name="mean smoothed", mode="lines",
                                      line=dict(color="black", width=5.0))
    trace_smoothed_means_upper = go.Scatter(x=years,
                                            y=smoothed_means+1.65*np.sqrt(smoothed_vars),
                                            mode="lines",
                                            line=dict(color="black",
                                                      width=0.5),
                                            showlegend=False)
    trace_smoothed_means_lower = go.Scatter(x=years,
                                            y=smoothed_means-1.65*np.sqrt(smoothed_vars),
                                            mode="lines",
                                            line=dict(color="black",
                                                      width=0.5),
                                            showlegend=False)
    fig.add_trace(trace_data)
    fig.add_trace(trace_smoothed_means)
    fig.add_trace(trace_smoothed_means_upper)
    fig.add_trace(trace_smoothed_means_lower)

    fig.update_yaxes(title="Nile River Annual Flow Volume at Aswan", range=(np.min(measurements), np.max(measurements)))
    fig.update_xaxes(title="Year")

    fig.show()

    fig = go.Figure()
    trace_smoothed_var = go.Scatter(x=years, y=smoothed_vars,
                                    name="smooth state variance", mode="lines",
                                    line=dict(color="black", width=1.0),
                                    showlegend=False)
    fig.add_trace(trace_smoothed_var)

    fig.update_yaxes(title="Smoothed State Variance", range=(5000, 17500))
    fig.update_xaxes(title="Year")

    fig.show()

    breakpoint()

if __name__ == "__main__":
   main(sys.argv)
