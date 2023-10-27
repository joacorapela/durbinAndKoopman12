
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append("../../src")
import tsa


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--a10", type=float, help="initial filter mean",
                        default=float("nan"))
    parser.add_argument("--P10", type=float, help="initial filter variance",
                        default=1e7)
    parser.add_argument("--sigma2Epsilon0", type=float,
                        help="observations noise variance",
                        default=15099.0)
    parser.add_argument("--sigma2Eta0", type=float,
                        help="state noise variance",
                        default=1469.1)
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../../data/Nile.csv")
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../../results/ch2/fig2_6{:s}.{:s}")

    args = parser.parse_args()
    a10 = args.a10
    P10 = args.P10
    sigma2Epsilon0 = args.sigma2Epsilon0
    sigma2Eta0 = args.sigma2Eta0
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    data = pd.read_csv(data_filename)
    years = data["time"].to_numpy()
    measurements = data["Nile"].to_numpy()

    measurements = np.append(measurements, np.ones(forecast_horizon)*np.nan)
    years = np.append(years, years[-1]+1+np.arange(forecast_horizon))

    llmLLcalc = LLMlogLikeCalc(yt=measurements)
    params0 = np.array([a10, P10, sigma2Epsilon0, sigma2Eta0])
    options = {"maxiter": maxiter, "disp": disp}
    def callback(intermediate_result):
        print("LL(a1={:.02f}, P1={:.02f}, lepsilon={:.02f}, leta={:.02f})={:.02f}".format(*intermediate_result.x, intermediate_result.fun})
    optim_res = scipy.optimize.minimize(fun=llmLLcalc.eval, x0=params0,
                                        method=method, jac=llmLLcal.grad,
                                        options=options, callback=callback,

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
