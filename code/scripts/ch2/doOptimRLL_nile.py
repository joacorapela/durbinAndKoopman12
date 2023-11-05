
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.optimize

sys.path.append("../../src")
import rllm


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--q0", type=float,
                        help="initial condition for q",
                        default=1.0)
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../../data/Nile.csv")
    parser.add_argument("--est_res_metadata_filename_pattern",
                        help="estimation metadata filename pattern", type=str,
                        default="../../../results/{:08d}_est_metadata.ini")
    parser.add_argument("--est_res_filename_pattern", type=str,
                        help="estimation result filename pattern",
                        default="../../../results/{:08d}_est_res.pickle")

    args = parser.parse_args()
    q0 = args.q0
    data_filename = args.data_filename
    est_res_metadata_filename_pattern = args.est_res_metadata_filename_pattern
    est_res_filename_pattern = args.est_res_filename_pattern

    data = pd.read_csv(data_filename)
    y = data["Nile"].to_numpy()

    maxiter = 100
    disp = True
    method = "BFGS"

    llmLLcalc = rllm.LogLikeCalculator(
        y=y)
    options = {"gtol": 1e-10, "maxiter": maxiter, "disp": disp}

    def callback(intermediate_result: scipy.optimize.OptimizeResult):
        print("LL(lq={:.02f})={:.06f}".format(
                  intermediate_result[0],
                  llmLLcalc.ll(intermediate_result),
              ))

    def fun(x):
        return -llmLLcalc.ll(x)

    params0 = [np.log(q0)]
    print("LL(lq={:.02f})={:.06f}".format(
        params0[0], llmLLcalc.ll(params0))
    )

    optim_res = scipy.optimize.minimize(fun=fun, x0=params0,
                                        method=method,
                                        options=options, callback=callback)
    print(optim_res)

    ll = -optim_res.fun
    q = np.exp(optim_res.x[0])

    llmKF = rllm.KalmanFilter(
            q=q,
    )
    filter_res = llmKF.predictBatch(y=y)
    a = filter_res[0]
    P = filter_res[1]
    N = len(y)
    F = np.array(P) + 1.0
    v = y - a
    s2ep = np.exp(np.log(1.0 / (N-1) * np.sum(v[1:]**2 / F[1:])))
    s2et = q * s2ep

    print(f"ll:   {ll}")
    print(f"q:    {q}")
    print(f"s2ep: {s2ep}")
    print(f"s2et: {s2et}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
