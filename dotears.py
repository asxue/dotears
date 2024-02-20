import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from sklearn.preprocessing import StandardScaler

import os
import argparse

def scale_data(data):
    scaler = StandardScaler()
    data_scaled = {}
    for k, v in data.items():
        scaled = scaler.fit_transform(v)
        data_scaled[k] = scaled
        
    return data_scaled

class DOTEARS:
    def __init__(self, data, lambda1=0.1, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16, scaled=False, w_threshold=0, obs_only=False):
        self.data = data
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.scaled = scaled
        self.obs_only = obs_only

#         self.p = data['obs'].shape[1]
        self.p = list(data.values())[0].shape[1]
        self.V_inverse = (self.estimate_exogenous_variances(data) ** (-1)) * np.identity(self.p)
        
        self.scaled = scaled
        if self.scaled:
#             self.original_variances = {}
#             for k, v in data.items():
#                 self.original_variances[k] = v.var(axis=0)
            self.data = scale_data(data)
    
    def estimate_exogenous_variances(self, data):
        p = list(data.values())[0].shape[1]
        variances = np.zeros(p)
        for k, v in data.items():
            if k == 'obs':
                continue
            variances[int(k)] = v.var(axis=0)[int(k)]

        return variances

    def loss(self, W):
        data = self.data
        p = self.p
        obs_only = self.obs_only

        V_inverse = self.V_inverse
        V_half = V_inverse ** 0.5

        if self.loss_type == 'l2':
            G_loss = 0
            loss = 0

            for j in data.keys():
                if self.obs_only:
                    if j != 'obs':
                        continue

                mask = np.ones((p, p))

                if j != 'obs':
                    mask[:, int(j)] = 0

                W_j = mask * W

                R = data[j] - data[j] @ W_j
#                 if self.scaled:
 #                    V_half = (V_inverse * self.original_variances[j]) ** 0.5

                # new gradient
                loss += 0.5 / data[j].shape[0] * ((R @ V_half) ** 2).sum()
#                 if self.scaled:
#                     G_loss += - 1.0 / data[j].shape[0] * data[j].T @ R @ (V_inverse * self.original_variances[j])
#                 else:
                G_loss += - 1.0 / data[j].shape[0] * data[j].T @ R @ V_inverse

            if not self.obs_only:
                loss /= len(data.keys())
                G_loss /= len(data.keys())
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(self, W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - self.p
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(self, w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        d = self.p
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(self, w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        rho = self.rho
        W = self._adj(w)
#         W = np.triu(W, 1) # added
        loss, G_loss = self.loss(W)
        h, G_h = self._h(W)
        obj = loss + 0.5 * rho * h * h + self.alpha * h + self.lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + self.alpha) * G_h
        g_obj = np.concatenate((G_smooth + self.lambda1, - G_smooth + self.lambda1), axis=None)
        return obj, g_obj

    def fit(self):
        d = self.p
        data = self.data
        w_est, self.rho, self.alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)

        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        for k, v in data.items():
            data[k] = data[k] - np.mean(data[k], axis=0, keepdims=True)

        for iter_number in range(self.max_iter):
            w_new, h_new = None, None
            while self.rho < self.rho_max:
                sol = sopt.minimize(self._func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = self._h(self._adj(w_new))

                if h_new > 0.25 * h:
                    self.rho *= 10
                else:
                    break
                    
                
            w_est, h = w_new, h_new
            self.alpha += self.rho * h
            if h <= self.h_tol or self.rho >= self.rho_max:
                break
        W_est = self._adj(w_est)
        W_est[np.abs(W_est) < self.w_threshold] = 0
        return W_est

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to input data file')
    parser.add_argument('--out', type=str, help='path for output')
    parser.add_argument('--lambda1', type=float, help='regularization parameter')
    args = parser.parse_args()

    data = dict(np.load(args.data))
    DOTEARS_obj = DOTEARS(data, lambda1=args.lambda1, scaled=False, w_threshold=0)
    w = DOTEARS_obj.fit()

    dirname = os.path.dirname(args.out)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save(args.out, w)
