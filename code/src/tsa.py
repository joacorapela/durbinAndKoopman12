
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
        vt = yt - at
        Ft = Pt + self._sigma2Epsilon
        Kt = Pt/Ft
        atgt = at + Kt * vt
        Ptgt = Pt*(1-Kt)
        return atgt, Ptgt, vt, Ft, Kt

    def smooth(self, ys, at, Pt, atgt, Ptgt):
        N = len(at)
        alphaHat = np.empty(N)
        Vt = np.empty(N)
        alphaHat[-1] = atgt[-1]
        Vt[-1] = Ptgt[-1]

        for i in range(N-2, -1, -1):
            alphaHat[i] = atgt[i] + Ptgt[i]/Pt[i+1] * (alphaHat[i+1] - at[i+1])
            Vt[i] = Ptgt[i] + (Ptgt[i]/Pt[i+1])**2 * (Vt[i+1] - Pt[i+1])
        return alphaHat, Vt


