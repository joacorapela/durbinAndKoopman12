
import sys
import argparse
import pickle
import numpy as np
import plotly.graph_objects as go

sys.path.append("../../src")
import tsa


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_res_number", type=int,
                        # help="simulation result number", default=372651)
                        help="simulation result number", default=27465920)
    parser.add_argument("--a1", type=float,
                        help="initial condition for initial state mean",
                        default=0.0)
    parser.add_argument("--P1", type=float,
                        help="initial condition for initial state variance",
                        default=1e2)
    parser.add_argument("--sigma_epsilon", type=float,
                        help=("initial condition for observations noise "
                              "variance"),
                        default=1.0)
    parser.add_argument("--sigma_eta", type=float,
                        help="initial condition for state noise variance",
                        default=1.0)
    parser.add_argument("--sim_res_filename_pattern", type=str,
                        help="simulation result filename pattern",
                        default="../../../results/{:08d}_simRes.pickle")
    parser.add_argument("--est_res_metadata_filename_pattern",
                        help="estimation metadata filename pattern", type=str,
                        default="../../../results/{:08d}_est_metadata.ini")
    parser.add_argument("--est_res_filename_pattern", type=str,
                        help="estimation result filename pattern",
                        default="../../../results/{:08d}_est_res.pickle")

    args = parser.parse_args()
    sim_res_number = args.sim_res_number
    a1 = args.a1
    lP1 = np.log(args.P1)
    ls2ep = np.log(args.sigma_epsilon**2)
    ls2et = np.log(args.sigma_eta**2)
    sim_res_filename_pattern = args.sim_res_filename_pattern

    sim_res_filename = sim_res_filename_pattern.format(sim_res_number)
    with open(sim_res_filename, "rb") as f:
        sim_res = pickle.load(f)
    y = sim_res["y"]

    llmLLcalc = tsa.LocalLevelModelLogLikeCalculator(y=y)

    a1s = np.linspace(-1.0, 1.0, 100)
    lP1s = np.log(np.linspace(1.0, 200.0, 100))
    ls2eps = np.log(np.linspace(0.1, 3.0, 100))
    ls2ets = np.log(np.linspace(0.1, 3.0, 100))

    lls = np.empty(len(ls2eps), dtype=np.double)
    for i, tls2ep in enumerate(ls2eps):
        lls[i] = llmLLcalc.ll(np.array([a1, lP1, tls2ep, ls2et]))

    fig = go.Figure()
    trace = go.Scatter(x=ls2eps, y=lls, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title_text="ls2ep")
    fig.update_yaxes(title_text="ll")
    fig.show()

    lls = np.empty(len(ls2eps), dtype=np.double)
    for i, tls2et in enumerate(ls2ets):
        lls[i] = llmLLcalc.ll(np.array([a1, lP1, ls2ep, tls2et]))

    fig = go.Figure()
    trace = go.Scatter(x=ls2ets, y=lls, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title_text="ls2et")
    fig.update_yaxes(title_text="ll")
    fig.show()

    lls = np.empty(len(a1s), dtype=np.double)
    for i, ta1 in enumerate(a1s):
        lls[i] = llmLLcalc.ll(np.array([ta1, lP1, ls2ep, ls2et]))

    fig = go.Figure()
    trace = go.Scatter(x=a1s, y=lls, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title_text="a1")
    fig.update_yaxes(title_text="ll")
    fig.show()

    lls = np.empty(len(a1s), dtype=np.double)
    for i, tlP1 in enumerate(lP1s):
        lls[i] = llmLLcalc.ll(np.array([a1, tlP1, ls2ep, ls2et]))

    fig = go.Figure()
    trace = go.Scatter(x=lP1s, y=lls, mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title_text="lP1")
    fig.update_yaxes(title_text="ll")
    fig.show()

#     lls = np.empty((len(a1s), len(lP1s)), dtype=np.double)
#     for i, ta1 in enumerate(a1s):
#         for j, tlP1 in enumerate(lP1s):
#             lls[i, j] = llmLLcalc.ll(np.array([ta1, tlP1, ls2ep, ls2et]))
# 
#     fig = go.Figure()
#     trace = go.Contour(z=lls, x=lP1s, y=a1s)
#     fig.add_trace(trace)
#     fig.update_xaxes(title_text="log P1")
#     fig.update_yaxes(title_text="log a1")
#     fig.show()
# 
#     lls = np.empty((len(ls2eps), len(ls2ets)), dtype=np.double)
#     for i, tls2ep in enumerate(ls2eps):
#         for j, tls2et in enumerate(ls2ets):
#             lls[i, j] = llmLLcalc.ll(np.array([a1, lP1, tls2ep, tls2et]))
# 
#     fig = go.Figure()
#     trace = go.Contour(z=lls, x=ls2ets, y=ls2eps)
#     fig.add_trace(trace)
#     fig.update_xaxes(title_text="ls2et")
#     fig.update_yaxes(title_text="ls2ep")
#     fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
