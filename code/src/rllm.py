import numpy as np


class KalmanFilter:
    def __init__(self, q):
        self._q = q

    def predict(self, atgt, Ptgt):
        atp1 = atgt
        Ptp1 = Ptgt + self._q
        return atp1, Ptp1

    def update(self, at, Pt, yt):
        Ft = Pt + 1.0
        vt = yt - at
        if not np.isnan(yt):
            Kt = Pt / Ft
            atgt = at + Kt * vt
        else:
            Kt = 0.0
            atgt = at
        Ptgt = Pt * (1.0 - Kt)
        return atgt, Ptgt, vt, Ft, Kt

    def predictBatch(self, y):
        N = len(y)
        ats = [np.nan for _ in range(N)]
        Pts = [np.nan for _ in range(N)]
        a2 = y[0]
        P2 = 1 + self._q
        at = a2
        Pt = P2
        for n, yt in enumerate(y[1:]):
            ats[n+1] = at
            Pts[n+1] = Pt
            atgt, Ptgt, _, _, _ = self.update(at=at, Pt=Pt, yt=yt)
            at, Pt = self.predict(atgt=atgt, Ptgt=Ptgt)
        return ats, Pts


class LogLikeCalculator:

    def __init__(self, y):
        self._y = y
        self._last_params = None

    def ll(self, params):
        lq = params[0]

        if self._last_params is None or \
           not np.array_equal(a1=self._last_params, a2=params):
            self._last_params = params.copy()
            self._update_filter_res(lq=lq)
        a = self._filter_res[0]
        P = self._filter_res[1]
        N = len(self._y)
        F = np.array(P) + 1.0
        v = self._y - a
        ls2ep = np.log(1.0 / (N-1) * np.sum(v[1:]**2 / F[1:]))
        ll = -0.5*(N * np.log(2*np.pi) + (N-1) / 2.0 +
                   (N-1) / 2.0 * ls2ep + np.sum(F[1:]))
        # print(f"params: {params[0]}")
        # print(f"ll: {ll}")
        return ll

    def _update_filter_res(self, lq):
        llmKF = KalmanFilter(
            q=np.exp(lq),
        )
        self._filter_res = llmKF.predictBatch(y=self._y)
