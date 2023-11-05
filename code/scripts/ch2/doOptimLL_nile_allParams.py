
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.optimize

sys.path.append("../../src")
import llm


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--a10", type=float,
                        help="initial condition for initial state mean",
                        default=0.0)
    parser.add_argument("--P10", type=float,
                        help="initial condition for initial state variance",
                        default=0.01)
    parser.add_argument("--sigma_epsilon0", type=float,
                        help=("initial condition for observations noise "
                              "variance"),
                        default=1.0)
    parser.add_argument("--sigma_eta0", type=float,
                        help="initial condition for state noise variance",
                        default=1.5)
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../../data/Nile.csv")
    parser.add_argument("--est_res_metadata_filename_pattern",
                        help="estimation metadata filename pattern", type=str,
                        default="../../../results/{:08d}_est_metadata.ini")
    parser.add_argument("--est_res_filename_pattern", type=str,
                        help="estimation result filename pattern",
                        default="../../../results/{:08d}_est_res.pickle")

    args = parser.parse_args()
    a10 = args.a10
    lP10 = np.log(args.P10)
    ls2ep0 = np.log(args.sigma_epsilon0**2)
    ls2et0 = np.log(args.sigma_eta0**2)
    data_filename = args.data_filename
    est_res_metadata_filename_pattern = args.est_res_metadata_filename_pattern
    est_res_filename_pattern = args.est_res_filename_pattern

    data = pd.read_csv(data_filename)
    y = data["Nile"].to_numpy()

    maxiter = 100
    disp = True
    method = "BFGS"

    params_to_estimate = ["a1", "lP1", "ls2ep", "ls2et"]
    params0 = [a10, lP10, ls2ep0, ls2et0]
    llmLLcalc = llm.LogLikeCalculator(
        y=y, params_to_estimate=params_to_estimate,
        fixed_params_values={})
    options = {"gtol": 1e-10, "maxiter": maxiter, "disp": disp}

    def callback(intermediate_result: scipy.optimize.OptimizeResult):
        print("LL(a1={:.02f}, lP1={:.02f}, ls2ep={:.02f}, "
              "ls2et={:.02f})".format(
                  intermediate_result[0],
                  intermediate_result[1],
                  intermediate_result[2],
                  intermediate_result[3],
              ))

    def fun(x):
        return -llmLLcalc.ll(x)

    def grad(x):
        return -llmLLcalc.grad(x)

    optim_res = scipy.optimize.minimize(fun=fun, x0=params0,
                                        method=method, jac=grad,
                                        options=options, callback=callback)
    print(optim_res)

    ll = -optim_res.fun
    a1 = optim_res.x[0]
    P1 = np.exp(optim_res.x[1])
    s2ep = np.exp(optim_res.x[2])
    s2et = np.exp(optim_res.x[3])
    F1 = P1 + s2ep
    v1 = y[0] - a1
    dll = ll + 0.5 * (np.log(F1) + v1**2 / F1)

    print(f"ll:   {ll}")
    print(f"dll:  {dll}")
    print(f"a1:   {a1}")
    print(f"P1:   {P1}")
    print(f"s2ep: {s2ep}")
    print(f"s2et: {s2et}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
