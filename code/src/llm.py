import numpy as np


class KalmanFilter:
    def __init__(self, sigma2Epsilon, sigma2Eta):
        self._sigma2Epsilon = sigma2Epsilon
        self._sigma2Eta = sigma2Eta

    def predict(self, atgt, Ptgt):
        atp1 = atgt
        Ptp1 = Ptgt + self._sigma2Eta
        return atp1, Ptp1

    def update(self, at, Pt, yt):
        Ft = Pt + self._sigma2Epsilon
        vt = yt - at
        if not np.isnan(yt):
            Kt = Pt / Ft
            atgt = at + Kt * vt
        else:
            Kt = 0.0
            atgt = at
        Ptgt = Pt * (1.0 - Kt)
        return atgt, Ptgt, vt, Ft, Kt

    def update(self, at, Pt, yt):
        Ft = Pt + self._sigma2Epsilon
        vt = yt - at
        if not np.isnan(yt):
            Kt = Pt / Ft
            atgt = at + Kt * vt
        else:
            Kt = 0.0
            atgt = at
        Ptgt = Pt * (1.0 - Kt)
        return atgt, Ptgt, vt, Ft, Kt

    def updateSS(self, at, Kss, yt):
        vt = yt - at
        if not np.isnan(yt):
            atgt = at + Kss * vt
        else:
            Kss = 0.0
            atgt = at
        return atgt, vt

    def smooth(self, at, Pt, atgt, Ptgt):
        N = len(at)
        alphaHat = np.empty(N)
        Vt = np.empty(N)
        alphaHat[-1] = atgt[-1]
        Vt[-1] = Ptgt[-1]

        for i in range(N-2, -1, -1):
            alphaHat[i] = atgt[i] + Ptgt[i]/Pt[i+1] * (alphaHat[i+1] - at[i+1])
            Vt[i] = Ptgt[i] + (Ptgt[i]/Pt[i+1])**2 * (Vt[i+1] - Pt[i+1])
        return alphaHat, Vt

    def predictBatch(self, y, a1, P1):
        N = len(y)
        ats = [np.nan for _ in range(N)]
        Pts = [np.nan for _ in range(N)]
        at = a1
        Pt = P1
        for n, yt in enumerate(y):
            ats[n] = at
            Pts[n] = Pt
            atgt, Ptgt, _, _, _ = self.update(at=at, Pt=Pt, yt=yt)
            at, Pt = self.predict(atgt=atgt, Ptgt=Ptgt)
        return ats, Pts

    def steadyStatePredictBatch(self, y, a1, P1, varSimThr=1e-4):
        q = self._sigma2Eta / self._sigma2Epsilon
        x = (q + np.sqrt(q**2 + 4*q)) / 2.0
        Pss = x * self._sigma2Epsilon
        Kss = Pss / (Pss + self._sigma2Epsilon)

        N = len(y)
        ats = [np.nan for _ in range(N)]
        Pts = [np.nan for _ in range(N)]
        at = a1
        Pt = P1
        t = 0
        while np.abs(Pt - Pss) > varSimThr and t < len(y):
            yt = y[t]
            ats[t] = at
            Pts[t] = Pt
            atgt, Ptgt, _, _, _ = self.update(at=at, Pt=Pt, yt=yt)
            at, Pt = self.predict(atgt=atgt, Ptgt=Ptgt)
            t += 1
        while t < len(y):
            yt = y[t]
            ats[t] = at
            Pts[t] = Pss
            at, _ = self.updateSS(at=at, Kss=Kss, yt=yt)
            t += 1
        return ats, Pts


class LogLikeCalculator:

    def __init__(self, y,
                 params_to_estimate=["a1", "lP1", "ls2ep", "ls2et"],
                 fixed_params_values={}):
        self._y = y
        self._last_params = None
        self._filter_res = None
        self._grad_filter_res = None
        self._params_to_estimate = params_to_estimate
        self._fixed_params_values = fixed_params_values

    def get_all_params(self, params):
        try:
            index = self._params_to_estimate.index("a1")
            a1 = params[index]
        except ValueError:
            a1 = self._fixed_params_values["a1"]

        try:
            index = self._params_to_estimate.index("lP1")
            lP1 = params[index]
        except ValueError:
            lP1 = self._fixed_params_values["lP1"]

        try:
            index = self._params_to_estimate.index("ls2ep")
            ls2ep = params[index]
        except ValueError:
            ls2ep = self._fixed_params_values["ls2ep"]

        try:
            index = self._params_to_estimate.index("ls2et")
            ls2et = params[index]
        except ValueError:
            ls2et = self._fixed_params_values["ls2et"]

        return a1, lP1, ls2ep, ls2et

    def ll(self, params):
        a1, lP1, ls2ep, ls2et = self.get_all_params(params=params)

        # s2ep = np.exp(ls2ep)
        # s2et = np.exp(ls2et)
        if self._last_params is None or \
           not np.array_equal(a1=self._last_params, a2=params):
            self._last_params = params.copy()
            self._update_filter_res(a1=a1, lP1=lP1, ls2ep=ls2ep, ls2et=ls2et)
        a = self._filter_res[0]
        P = self._filter_res[1]
        N = len(self._y)
        F = P + np.exp(ls2ep)
        ll = -0.5*(N * np.log(2*np.pi) +
                   np.sum(np.log(F) + (self._y - a)**2 / F))
        # print(f"params: {params[0]}")
        # print(f"ll: {ll}")
        return ll

    def grad(self, params):
        a1, lP1, ls2ep, ls2et = self.get_all_params(params=params)
        s2ep = np.exp(ls2ep)
        # s2et = np.exp(ls2et)
        if self._last_params is None or \
           not np.array_equal(a1=self._last_params, a2=params):
            self._last_params = params.copy()
            self._update_filter_res(a1=a1, lP1=lP1, ls2ep=ls2ep, ls2et=ls2et)
        if self._grad_filter_res is None:
            self._update_grad_filter_res(a1=a1, lP1=lP1, ls2ep=ls2ep,
                                         ls2et=ls2et)
        a = self._filter_res[0]
        P = self._filter_res[1]
        v = self._y - a
        F = P + s2ep
        pPtWRTlP1   = self._grad_filter_res[0, 0, :]
        pPtWRTa1    = self._grad_filter_res[0, 1, :]
        pPtWRTls2ep = self._grad_filter_res[0, 2, :]
        pPtWRTls2et = self._grad_filter_res[0, 3, :]
        patWRTlP1   = self._grad_filter_res[1, 0, :]
        patWRTa1    = self._grad_filter_res[1, 1, :]
        patWRTls2ep = self._grad_filter_res[1, 2, :]
        patWRTls2et = self._grad_filter_res[1, 3, :]
        grad_dict = {}
        # glP1
        if "lP1" in self._params_to_estimate:
            grad_dict["lP1"] = -0.5 * np.sum(
                (pPtWRTlP1 / F +
                 (-2 * v * patWRTlP1 * F -
                  v**2 * pPtWRTlP1) / F**2)
            )
        # ga1
        if "a1" in self._params_to_estimate:
            grad_dict["a1"] = -0.5 * np.sum(
                (pPtWRTa1 / F +
                 (-2 * v * patWRTa1 * F -
                  v**2 * pPtWRTa1) / F**2)
            )
        # gls2ep
        if "ls2ep" in self._params_to_estimate:
            grad_dict["ls2ep"] = -0.5 * np.sum(
                ((pPtWRTls2ep + s2ep) / F +
                 (-2 * v * patWRTls2ep * F -
                  v**2 * (pPtWRTls2ep + s2ep)) / F**2)
            )
        # gls2et
        if "ls2et" in self._params_to_estimate:
            grad_dict["ls2et"] = -0.5 * np.sum(
                (pPtWRTls2et / F +
                 (-2 * v * patWRTls2et * F -
                  v**2 * pPtWRTls2et) / F**2)
            )

        grad_list = []
        for param_name in self._params_to_estimate:
            grad_list.append(grad_dict[param_name])
        grad = np.array(grad_list)
        # print(f"params: {params[0]}")
        # print(f"grad: {grad[0]}")
        return grad

    def _update_filter_res(self, a1, lP1, ls2ep, ls2et):
        llmKF = KalmanFilter(
            sigma2Epsilon=np.exp(ls2ep),
            sigma2Eta=np.exp(ls2et),
        )
        self._filter_res = llmKF.predictBatch(y=self._y, a1=a1, P1=np.exp(lP1))
        self._grad_filter_res = None

    def _update_grad_filter_res(self, a1, lP1, ls2ep, ls2et):
        T = len(self._y)
        a = self._filter_res[0]
        P = self._filter_res[1]
        F = P + np.exp(ls2et)
        K = P / F
        v = self._y - a

        self._grad_filter_res = np.empty((2, 4, T), dtype=np.double)
        pPtWRTlP1 = self._grad_filter_res[0, 0, :]
        pPtWRTa1  = self._grad_filter_res[0, 1, :]
        pPtWRTls2ep = self._grad_filter_res[0, 2, :]
        pPtWRTls2et = self._grad_filter_res[0, 3, :]
        patWRTlP1 = self._grad_filter_res[1, 0, :]
        patWRTa1  = self._grad_filter_res[1, 1, :]
        patWRTls2ep = self._grad_filter_res[1, 2, :]
        patWRTls2et = self._grad_filter_res[1, 3, :]

        grad_F = np.empty((4, T), dtype=np.double)
        pFtWRTlP1 = grad_F[0, :]
        pFtWRTa1 = grad_F[1, :]
        pFtWRTls2ep = grad_F[2, :]
        pFtWRTls2et = grad_F[3, :]

        grad_K = np.empty((4, T), dtype=np.double)
        pKtWRTlP1 = grad_K[0, :]
        pKtWRTa1 = grad_K[1, :]
        pKtWRTls2ep = grad_K[2, :]
        pKtWRTls2et = grad_K[3, :]

        pPtWRTlP1[0] = np.exp(lP1)
        pPtWRTa1[0] = 0.0
        pPtWRTls2ep[0] = 0.0
        pPtWRTls2et[0] = 0.0
        patWRTlP1[0] = 0.0
        patWRTa1[0] = 1.0
        patWRTls2ep[0] = 0.0
        patWRTls2et[0] = 0.0

        pKtWRTlP1[0] = np.exp(lP1) * np.exp(ls2ep) / \
            (np.exp(lP1) + np.exp(ls2ep))**2
        pKtWRTa1[0] = 0.0
        pKtWRTls2ep[0] = -np.exp(lP1) * np.exp(ls2ep) / \
            (np.exp(lP1) + np.exp(ls2ep))**2
        pKtWRTls2et[0] = 0.0

        pFtWRTlP1[0] = np.exp(lP1)
        pFtWRTa1[0] = 0.0
        pFtWRTls2ep[0] = np.exp(ls2ep)
        pFtWRTls2et[0] = 0.0

        for t in range(1, T):
            # grad Pt wrt lP1
            pPtWRTlP1[t] = \
                pPtWRTlP1[t-1] * (1 - K[t-1]) - \
                P[t-1] * pKtWRTlP1[t-1]
            # grad Pt wrt a1
            pPtWRTa1[t] = \
                pPtWRTa1[t-1] * (1 - K[t-1]) - \
                P[t-1] * pKtWRTa1[t-1]
            # grad Pt wrt ls2ep
            pPtWRTls2ep[t] = \
                pPtWRTls2ep[t-1] * (1 - K[t-1]) - \
                P[t-1] * pKtWRTls2ep[t-1]
            # grad Pt wrt ls2et
            pPtWRTls2et[t] = \
                pPtWRTls2et[t-1] * (1 - K[t-1]) - \
                P[t-1] * pKtWRTls2et[t-1] + np.exp(ls2et)

            # grad at wrt lP1
            patWRTlP1[t] = \
                patWRTlP1[t-1] * (1 - K[t-1]) + \
                pKtWRTlP1[t-1] * v[t-1]
            # grad at wrt a1
            patWRTa1[t] = \
                patWRTa1[t-1] * (1 - K[t-1]) + \
                pKtWRTa1[t-1] * v[t-1]
            # grad at wrt ls2ep
            patWRTls2ep[t] = \
                patWRTls2ep[t-1] * (1 - K[t-1]) + \
                pKtWRTls2ep[t-1] * v[t-1]
            # grad at wrt ls2et
            patWRTls2et[t] = \
                patWRTls2et[t-1] * (1 - K[t-1]) + \
                pKtWRTls2et[t-1] * v[t-1]

            # grad Ft wrt lP1
            pFtWRTlP1[t] = pPtWRTlP1[t]
            # grad Ft wrt a1
            pFtWRTa1[t] = pPtWRTa1[t]
            # grad Ft wrt ls2ep
            pFtWRTls2ep[t] = pPtWRTls2ep[t] + np.exp(ls2ep)
            # grad Ft wrt ls2et
            pFtWRTls2et[t] = pPtWRTls2et[t]

            # grad Kt wrt lP1
            pKtWRTlP1[t] = (pPtWRTlP1[t] * F[t] -
                            P[t] * pFtWRTlP1[t]) / F[t]**2
            # grad Kt wrt a1
            pKtWRTa1[t] = (pPtWRTa1[t] * F[t] -
                           P[t] * pFtWRTa1[t]) / F[t]**2
            # grad Kt wrt ls2ep
            pKtWRTls2ep[t] = (pPtWRTls2ep[t] * F[t] -
                              P[t] * pFtWRTls2ep[t]) / F[t]**2
            # grad Kt wrt ls2et
            pKtWRTls2et[t] = (pPtWRTls2et[t] * F[t] -
                              P[t] * pFtWRTls2et[t]) / F[t]**2


class Simulator:

    def simulate(self, sigma_epsilon, sigma_eta, a1, P1, N):
        eta = np.random.normal(loc=0, scale=sigma_eta, size=N)
        epsilon = np.random.normal(loc=0, scale=sigma_epsilon, size=N)
        alpha = np.empty(shape=N)
        alpha[:] = np.nan
        y = np.empty(shape=N)
        y[:] = np.nan

        alpha[0] = np.random.normal(loc=a1, scale=np.sqrt(P1), size=1)
        for n in range(N-1):
            y[n] = alpha[n] + epsilon[n]
            alpha[n + 1] = alpha[n] + eta[n]
        y[-1] = alpha[-1] + epsilon[-1]
        return y, alpha
