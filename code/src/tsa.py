
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
            Kt = Pt/Ft
            atgt = at + Kt * vt
        else:
            Kt = 0.0
            atgt = at
        Ptgt = Pt * (1-Kt)
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
        ats = [None] * N
        Pts = [None] * N
        at = a1
        Pt = P1
        for n, yt in enumerate(y):
            ats[n] = at
            Pts[n] = Pt
            atgt, Ptgt, _, _, _ = self.update(at=at, Pt=Pt, yt=yt)
            at, Pt = self.predict(atgt, Ptgt):
        return ats, Pts

class LocalLevelModelLogLikeCalculator:

    def __init__(y):
        self._y = y
        self._last_params = None
        self._filter_res = None
        self._grad_filter_res = None

    def ll(params):
        # params = [P1,a1,lep,let]
        a1  = params[0]
        lP1 = params[1]
        lep = params[2]
        let = params[3]
        s2ep = np.exp(lep)
        s2et = np.exp(let)
        if self._last_params is None or \
           not np.array_equal(a1=self._last_params, a2=params):
            self._last_params = params.copy()
            self._update_filter_res(s2ep=s2ep, s2et=s2et, a1=a1,
                                    P1=np.exp(lP1))
        ats, Pts = *self._filter_res
        N = len(self._y)
        aux = Pts + s2ep 
        ll = 0.5*(-N * np.log(2*np.pi) -
                  np.sum(np.log(aux) - (self._y - ats)**2/aux))
        return ll

    def grad(params):
        # params = [P1,a1,lep,let]
        a1  = params[0]
        lP1 = params[1]
        lep = params[2]
        let = params[3]
        s2ep = np.exp(lep)
        s2et = np.exp(let)
        if self._last_params is None or \
           not np.array_equal(a1=self._last_params, a2=params):
            self._last_params = params.copy()
            self._update_filter_res(lep=lep, let=let, a1=a1, lP1=lP1)
        if self._grad_filter_res is None:
            self._update_grad_filter_res(lep=lep, let=let, a1=a1, lP1=lP1)
        a = self._filter_res[0]
        P = self._filter_res[0]
        v = y - a
        # glP1
        glP1 = -0.5 * (self._grad_filter_res[0, 0, :] / (P + s2e) -
                       (-2 * v * self._grad_filter_res[1, 0, :] * (P + s2ep) -
                        self._grad_filter_res[0, 0, :]) / (P + s2ep)**2)
        # gla1
        gla1 = -0.5 * (self._grad_filter_res[0, 1, :] / (P + s2e) -
                       (-2 * v * self._grad_filter_res[1, 1, :] * (P + s2ep) -
                        self._grad_filter_res[0, 1, :]) / (P + s2ep)**2)
        # glep
        glep = -0.5 * ((self._grad_filter_res[0, 2, :] + s2ep) / (P + s2ep) -
                       (-2 * v * self._grad_filter_res[1, 2, :] * (P + s2ep) - 
                        v**2 * (self._grad_filter_res[0, 2, :] + sigma2ep)) / (P + s2ep)**2)
        # glet
        glet = -0.5 * (self._grad_filter_res[0, 3, :] / (P + s2ep) -
                       (-2 * v * self._grad_filter_res[1, 3, :] * (P + s2ep) -
                        v**2 * self._grad_filter_res[0, 3, :]) / (P + s2ep)**2)
        return


    def _update_filter_res(lep, let, a1, lP1):
        llmKF = LocalLevelModelKalmanFilter(
            sigma2Epsilon=np.exp(lep),
            sigma2Eta=np.exp(let),
        )
        self._filter_res = llmKF.predictBatch(y=self.y, a1=a1, P1=np.exp(lP1))
        self._grad_filter_res = None

    def _update_grad_filter_res(lep, let, a1, lP1):
        T = len(self._y)
        as = self._filter_res[0]
        Ps = self._filter_res[1]
        Fs = Ps + np.exp(let)
        Ks = Ps / Fs

        # partialPtWRTlP1 = self._grad_filter_res[0, 0, :]
        # partialPtWRTa1  = self._grad_filter_res[0, 1, :]
        # partialPtWRTlep = self._grad_filter_res[0, 2, :]
        # partialPtWRTlet = self._grad_filter_res[0, 3, :]
        # partialatWRTlP1 = self._grad_filter_res[1, 0, :]
        # partialatWRTa1  = self._grad_filter_res[1, 1, :]
        # partialatWRTlep = self._grad_filter_res[1, 2, :]
        # partialatWRTlet = self._grad_filter_res[1, 3, :]
        self._grad_filter_res = np.empty((2, 4, T), dtype=np.double)

        grad_Fs = np.empty((4, T), dtype=np.double)
        # partialFtWRTlP1 = Fs[0, :]
        # partialFtWRTla1 = Fs[1, :]
        # partialFtWRTlep = Fs[2, :]
        # partialFtWRTlet = Fs[3, :]
        grad_Ks = np.empty((4, T), dtype=np.double)
        # partialKtWRTlP1 = Ks[0, :]
        # partialKtWRTla1 = Ks[1, :]
        # partialKtWRTlep = Ks[2, :]
        # partialKtWRTlet = Ks[3, :]

        self._grad_filter_res[0, 0, 0] = np.exp(lP1)
        self._grad_filter_res[0, 1, 0] = 0.0
        self._grad_filter_res[0, 2, 0] = 0.0
        self._grad_filter_res[0, 3, 0] = 0.0
        self._grad_filter_res[1, 0, 0] = 0.0
        self._grad_filter_res[1, 1, 0] = 1.0
        self._grad_filter_res[1, 2, 0] = 0.0
        self._grad_filter_res[1, 3, 0] = 0.0

        grad_Ks[0, 0] =  np.exp(lP1) * np.exp(lep) / (np.exp(lP1) + np.exp(lep))**2
        grad_Ks[1, 0] =  0.0
        grad_Ks[2, 0] = -np.exp(lP1) * np.exp(lep) / (np.exp(lP1) + np.exp(lep))**2
        grad_Ks[3, 0] =  0.0

        grad_Fs[0, 0] = np.exp(lP1)
        grad_Fs[1, 0] = 0.0
        grad_Fs[2, 0] = np.exp(lep)
        grad_Fs[3, 0] = 0.0

        for t in range(1, T):
            # grad Pt wrt lP1
            self._grad_filter_res[0, 0, t] = \
                self._grad_filter_res[0, 0, t-1] * (1 - Ks[t-1]) - \
                Ps[t-1] * grad_Ks[0, t-1]
            # grad Pt wrt a1
            self._grad_filter_res[0, 1, t] = \
                self._grad_filter_res[0, 1, t-1] * (1 - Ks[t-1]) - \
                Ps[t-1] * grad_Ks[1, t-1]
            # grad Pt wrt lep
            self._grad_filter_res[0, 2, t] = \
                self._grad_filter_res[0, 2, t-1] * (1 - Ks[t-1]) - \
                Ps[t-1] * grad_Ks[2, t-1]
            # grad Pt wrt let
            self._grad_filter_res[0, 3, t] = \
                self._grad_filter_res[0, 3, t-1] * (1 - Ks[t-1]) - \
                Ps[t-1] * grad_Ks[3, t-1] + np.exp(let)

            # grad at wrt lP1
            self._grad_filter_res[1, 0, t] = \
                self._grad_filter_res[1, 0, t-1] * (1 - Ks[t-1]) + \
                grad_Ks[0, t-1] * (self.y[t-1] * as[t-1])
            # grad at wrt a1
            self._grad_filter_res[1, 1, t] = \
                self._grad_filter_res[1, 1, t-1] * (1 - Ks[t-1]) + \
                grad_Ks[1, t-1] * (self.y[t-1] * as[t-1])
            # grad at wrt lep
            self._grad_filter_res[1, 2, t] = \
                self._grad_filter_res[1, 2, t-1] * (1 - Ks[t-1]) + \
                grad_Ks[2, t-1] * (self.y[t-1] * as[t-1])
            # grad at wrt let
            self._grad_filter_res[1, 3, t] = \
                self._grad_filter_res[1, 3, t-1] * (1 - Ks[t-1]) + \
                grad_Ks[3, t-1] * (self.y[t-1] * as[t-1])

            # grad Ft wrt lP1
            grad_Fs[0, t] = self._grad_filter_res[0, 0, t]
            # grad Ft wrt a1
            grad_Fs[1, t] = self._grad_filter_res[0, 1, t]
            # grad Ft wrt lep
            grad_Fs[2, t] = self._grad_filter_res[0, 2, t] + np.exp(lep)
            # grad Ft wrt let
            grad_Fs[3, t] = self._grad_filter_res[0, 3, t]

            # grad Kt wrt lP1
            grad_Ks[0, t] = (self._grad_filter_res[0, 0, t] * Fs[t] - \
                             Ps[t] * grad_Fs[0, t]) / Fs[t]**2
            # grad Kt wrt a1
            grad_Ks[1, t] = (self._grad_filter_res[0, 1, t] * Fs[t] - \
                             Ps[t] * grad_Fs[1, t]) / Fs[t]**2
            # grad Kt wrt lep
            grad_Ks[2, t] = (self._grad_filter_res[0, 2, t] * Fs[t] - \
                             Ps[t] * grad_Fs[2, t]) / Fs[t]**2
            # grad Kt wrt let
            grad_Ks[3, t] = (self._grad_filter_res[0, 3, t] * Fs[t] - \
                             Ps[t] * grad_Fs[3, t]) / Fs[t]**2

