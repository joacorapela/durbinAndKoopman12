
import sys
import argparse
import pickle
import numpy as np
import scipy.optimize

sys.path.append("../../src")
import tsa


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_res_number", type=int,
                        help="simulation result number", default=16347168)
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
    a10 = args.a10
    lP10 = np.log(args.P10)
    ls2ep0 = np.log(args.sigma_epsilon0**2)
    ls2et0 = np.log(args.sigma_eta0**2)
    sim_res_filename_pattern = args.sim_res_filename_pattern
    est_res_metadata_filename_pattern = args.est_res_metadata_filename_pattern
    est_res_filename_pattern = args.est_res_filename_pattern

    maxiter = 100
    disp = True
    method = "BFGS"

    sim_res_filename = sim_res_filename_pattern.format(sim_res_number)
    with open(sim_res_filename, "rb") as f:
        sim_res = pickle.load(f)
    y = sim_res["y"]

    params_to_estimate = ["ls2ep", "ls2et"]
    params0 = [ls2ep0, ls2et0]
    llmLLcalc = tsa.LocalLevelModelLogLikeCalculator(
        y=y, params_to_estimate=params_to_estimate,
        fixed_params_values={"a1": a10, "lP1": lP10})
    options = {"gtol": 1e-10, "maxiter": maxiter, "disp": disp}

    def callback(intermediate_result: scipy.optimize.OptimizeResult):
        # print("LL(a1={:.02f}, lP1={:.02f}, ls2ep={:.02f}, ls2eta={:.02f})={:.02f}".format(
        #     intermediate_result["x"][0],
        #     intermediate_result["x"][1],
        #     intermediate_result["x"][2],
        #     intermediate_result["x"][3],
        #     intermediate_result["fun"]))
        # print("LL(ls2ep={:.02f}, ls2et={:.02f})".format(
        #     intermediate_result[0],
        #     intermediate_result[1]))
        print("LL(ls2ep={:.02f}, ls2et={:.02f})".format(
            intermediate_result[0],
            intermediate_result[1],
        ))

    def fun(x):
        return -llmLLcalc.ll(x)

    def grad(x):
        return -llmLLcalc.grad(x)

    optim_res = scipy.optimize.minimize(fun=fun, x0=params0,
                                        # method=method, jac=grad,
                                        method="BFGS",
                                        options=options, callback=callback)
    print(optim_res)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
