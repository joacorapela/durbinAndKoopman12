
import sys
import os
import pickle
import argparse
import configparser
import numpy as np
import plotly.graph_objects as go

sys.path.append("../../src")
import llm


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int,
                        help="number of simulated observations",
                        default=1000)
    parser.add_argument("--a1", type=float, help="initial state mean",
                        default=0.0)
    parser.add_argument("--P1", type=float, help="initial state variance",
                        default=1e-2)
    parser.add_argument("--sigma_epsilon", type=float,
                        help="observations noise variance",
                        default=1.0)
    parser.add_argument("--sigma_eta", type=float,
                        help="state noise variance",
                        default=1.0)
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename",
                        default="../../../figures/{:08d}_simulation.{:s}")
    parser.add_argument("--sim_res_config_filename_pattern",
                        help="Simulation result configuration filename "
                             "pattern", type=str,
                        default="../../../results/{:08d}_simulation_metaData.ini")
    parser.add_argument("--sim_res_filename_pattern", type=str,
                        help="simulation result filename pattern",
                        default="../../../results/{:08d}_simRes.pickle")

    args = parser.parse_args()
    N = args.N
    a1 = args.a1
    P1 = args.P1
    sigma_epsilon = args.sigma_epsilon
    sigma_eta = args.sigma_eta
    fig_filename_pattern = args.fig_filename_pattern
    sim_res_config_filename_pattern = args.sim_res_config_filename_pattern
    sim_res_filename_pattern = args.sim_res_filename_pattern

    llm_simulator = llm.Simulator()
    y, alpha = llm_simulator.simulate(sigma_epsilon=sigma_epsilon,
                                      sigma_eta=sigma_eta, a1=a1, P1=P1, N=N)
    sim_res = {"alpha": alpha, "y": y}
    samples = np.arange(len(y))

    random_prefix_used = True
    while random_prefix_used:
        sim_number = np.random.randint(0, 10**8)
        sim_res_config_filename = sim_res_config_filename_pattern.format(
            sim_number)
        if not os.path.exists(sim_res_config_filename):
            random_prefix_used = False
    sim_res_filename = sim_res_filename_pattern.format(sim_number)

    sim_res_config = configparser.ConfigParser()
    sim_res_config["simulation_params"] = {"N":N, "a1": a1, "P1": P1,
                                           "sigma_epsilon": sigma_epsilon,
                                           "sigma_eta": sigma_eta}
    sim_res_config["simulation_results"] = {"sim_res_filename":
                                            sim_res_filename}
    with open(sim_res_config_filename, "w") as f:
        sim_res_config.write(f)

    with open(sim_res_filename, "wb") as f:
        pickle.dump(sim_res, f)

    print(f"Simulation results filename: {sim_res_filename}")

    fig = go.Figure()
    trace = go.Scatter(x=samples, y=alpha, name="state", mode="lines+markers")
    fig.add_trace(trace)
    trace = go.Scatter(x=samples, y=y, name="observation", mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title_text="Sample")

    static_fig_filename = fig_filename_pattern.format(sim_number, "png")
    dynamic_fig_filename = fig_filename_pattern.format(sim_number, "html")
    fig.write_image(static_fig_filename)
    fig.write_html(dynamic_fig_filename)
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
