
import numpy as np


class LocalLevelModelKalmanFilter:
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
        ats = [None for _ in range(N)]
        Pts = [None for _ in range(N)]
        at = a1
        Pt = P1
        for n, yt in enumerate(y):
            ats[n] = at
            Pts[n] = Pt
            atgt, Ptgt, _, _, _ = self.update(at=at, Pt=Pt, yt=yt)
            at, Pt = self.predict(atgt=atgt, Ptgt=Ptgt)
        return ats, Pts


class LocalLevelModelLogLikeCalculator:

    def __init__(self, y):
        self._y = y
        self._last_params = None
        self._filter_res = None
        self._grad_filter_res = None

    def ll(self, params):
        # params = [P1,a1,ls2ep,ls2et]
        a1 = params[0]
        lP1 = params[1]
        ls2ep = params[2]
        ls2et = params[3]
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
        return ll

    def grad(self, params):
        # params = [P1,a1,ls2ep,ls2et]
        a1 = params[0]
        lP1 = params[1]
        ls2ep = params[2]
        ls2et = params[3]
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
        # glP1
        glP1 = np.sum(
            -0.5 * (self._grad_filter_res[0, 0, :] / (P + s2ep) -
                    (-2 * v * self._grad_filter_res[1, 0, :] * (P + s2ep) -
                     self._grad_filter_res[0, 0, :]) / (P + s2ep)**2)
        )
        # gla1
        gla1 = np.sum(
            -0.5 * (self._grad_filter_res[0, 1, :] / (P + s2ep) -
                    (-2 * v * self._grad_filter_res[1, 1, :] * (P + s2ep) -
                     self._grad_filter_res[0, 1, :]) / (P + s2ep)**2)
        )
        # gls2ep
        gls2ep = np.sum(
            -0.5 * ((self._grad_filter_res[0, 2, :] + s2ep) / (P + s2ep) -
                    (-2 * v * self._grad_filter_res[1, 2, :] * (P + s2ep) -
                     v**2 * (self._grad_filter_res[0, 2, :] + s2ep)) /
                    (P + s2ep)**2)
        )
        # gls2et
        gls2et = np.sum(
            -0.5 * (self._grad_filter_res[0, 3, :] / (P + s2ep) -
                    (-2 * v * self._grad_filter_res[1, 3, :] * (P + s2ep) -
                     v**2 * self._grad_filter_res[0, 3, :]) / (P + s2ep)**2)
        )
        grad = np.array([glP1, gla1, gls2ep, gls2et])
        return grad

    def _update_filter_res(self, a1, lP1, ls2ep, ls2et):
        llmKF = LocalLevelModelKalmanFilter(
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

        # partialPtWRTlP1 = self._grad_filter_res[0, 0, :]
        # partialPtWRTa1  = self._grad_filter_res[0, 1, :]
        # partialPtWRTls2ep = self._grad_filter_res[0, 2, :]
        # partialPtWRTls2et = self._grad_filter_res[0, 3, :]
        # partialatWRTlP1 = self._grad_filter_res[1, 0, :]
        # partialatWRTa1  = self._grad_filter_res[1, 1, :]
        # partialatWRTls2ep = self._grad_filter_res[1, 2, :]
        # partialatWRTls2et = self._grad_filter_res[1, 3, :]
        self._grad_filter_res = np.empty((2, 4, T), dtype=np.double)

        grad_F = np.empty((4, T), dtype=np.double)
        # partialFtWRTlP1 = F[0, :]
        # partialFtWRTla1 = F[1, :]
        # partialFtWRTls2ep = F[2, :]
        # partialFtWRTls2et = F[3, :]
        grad_K = np.empty((4, T), dtype=np.double)
        # partialKtWRTlP1 = K[0, :]
        # partialKtWRTla1 = K[1, :]
        # partialKtWRTls2ep = K[2, :]
        # partialKtWRTls2et = K[3, :]

        self._grad_filter_res[0, 0, 0] = np.exp(lP1)
        self._grad_filter_res[0, 1, 0] = 0.0
        self._grad_filter_res[0, 2, 0] = 0.0
        self._grad_filter_res[0, 3, 0] = 0.0
        self._grad_filter_res[1, 0, 0] = 0.0
        self._grad_filter_res[1, 1, 0] = 1.0
        self._grad_filter_res[1, 2, 0] = 0.0
        self._grad_filter_res[1, 3, 0] = 0.0

        grad_K[0, 0] = (np.exp(lP1) * np.exp(ls2ep) /
                        (np.exp(lP1) + np.exp(ls2ep))**2)
        grad_K[1, 0] = 0.0
        grad_K[2, 0] = (-np.exp(lP1) * np.exp(ls2ep) /
                        (np.exp(lP1) + np.exp(ls2ep)**2))
        grad_K[3, 0] = 0.0

        grad_F[0, 0] = np.exp(lP1)
        grad_F[1, 0] = 0.0
        grad_F[2, 0] = np.exp(ls2ep)
        grad_F[3, 0] = 0.0

        for t in range(1, T):
            # grad Pt wrt lP1
            self._grad_filter_res[0, 0, t] = \
                self._grad_filter_res[0, 0, t-1] * (1 - K[t-1]) - \
                P[t-1] * grad_K[0, t-1]
            # grad Pt wrt a1
            self._grad_filter_res[0, 1, t] = \
                self._grad_filter_res[0, 1, t-1] * (1 - K[t-1]) - \
                P[t-1] * grad_K[1, t-1]
            # grad Pt wrt ls2ep
            self._grad_filter_res[0, 2, t] = \
                self._grad_filter_res[0, 2, t-1] * (1 - K[t-1]) - \
                P[t-1] * grad_K[2, t-1]
            # grad Pt wrt ls2et
            self._grad_filter_res[0, 3, t] = \
                self._grad_filter_res[0, 3, t-1] * (1 - K[t-1]) - \
                P[t-1] * grad_K[3, t-1] + np.exp(ls2et)

            # grad at wrt lP1
            self._grad_filter_res[1, 0, t] = \
                self._grad_filter_res[1, 0, t-1] * (1 - K[t-1]) + \
                grad_K[0, t-1] * (self._y[t-1] * a[t-1])
            # grad at wrt a1
            self._grad_filter_res[1, 1, t] = \
                self._grad_filter_res[1, 1, t-1] * (1 - K[t-1]) + \
                grad_K[1, t-1] * (self._y[t-1] * a[t-1])
            # grad at wrt ls2ep
            self._grad_filter_res[1, 2, t] = \
                self._grad_filter_res[1, 2, t-1] * (1 - K[t-1]) + \
                grad_K[2, t-1] * (self._y[t-1] * a[t-1])
            # grad at wrt ls2et
            self._grad_filter_res[1, 3, t] = \
                self._grad_filter_res[1, 3, t-1] * (1 - K[t-1]) + \
                grad_K[3, t-1] * (self._y[t-1] * a[t-1])

            # grad Ft wrt lP1
            grad_F[0, t] = self._grad_filter_res[0, 0, t]
            # grad Ft wrt a1
            grad_F[1, t] = self._grad_filter_res[0, 1, t]
            # grad Ft wrt ls2ep
            grad_F[2, t] = self._grad_filter_res[0, 2, t] + np.exp(ls2ep)
            # grad Ft wrt ls2et
            grad_F[3, t] = self._grad_filter_res[0, 3, t]

            # grad Kt wrt lP1
            grad_K[0, t] = (self._grad_filter_res[0, 0, t] * F[t] -
                            P[t] * grad_F[0, t]) / F[t]**2
            # grad Kt wrt a1
            grad_K[1, t] = (self._grad_filter_res[0, 1, t] * F[t] -
                            P[t] * grad_F[1, t]) / F[t]**2
            # grad Kt wrt ls2ep
            grad_K[2, t] = (self._grad_filter_res[0, 2, t] * F[t] -
                            P[t] * grad_F[2, t]) / F[t]**2
            # grad Kt wrt ls2et
            grad_K[3, t] = (self._grad_filter_res[0, 3, t] * F[t] -
                            P[t] * grad_F[3, t]) / F[t]**2


class LocalLevelModelSimulator:

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
