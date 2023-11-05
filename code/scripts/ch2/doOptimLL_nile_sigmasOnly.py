
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.optimize

sys.path.append("../../src")
import llm


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--P1", type=float,
                        help="state variance",
                        default=1e10)
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
    P1 = args.P1
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

    a1 = y[0]
    lP1 = np.log(P1)

    params_to_estimate = ["ls2ep", "ls2et"]
    params0 = [ls2ep0, ls2et0]
    llmLLcalc = llm.LogLikeCalculator(
        y=y, params_to_estimate=params_to_estimate,
        fixed_params_values={"a1": a1, "lP1": lP1})
    options = {"gtol": 1e-10, "maxiter": maxiter, "disp": disp}

    def callback(intermediate_result: scipy.optimize.OptimizeResult):
        print("LL(ls2ep={:.02f}, ls2et={:.02f})={:.04f}".format(
                  intermediate_result[0],
                  intermediate_result[1],
                  llmLLcalc.ll(intermediate_result),
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
    s2ep = np.exp(optim_res.x[0])
    s2et = np.exp(optim_res.x[1])
    F1 = P1 + s2ep
    v1 = y[0] - a1
    dll = ll + 0.5 * (np.log(F1) + v1**2 / F1)

    print(f"ll:   {ll}")
    print(f"dll:  {dll}")
    print(f"s2ep: {s2ep}")
    print(f"s2et: {s2et}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
