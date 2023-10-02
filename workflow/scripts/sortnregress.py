import numpy as np
from sklearn.linear_model import LinearRegression, LassoLarsIC

from dotears import scale_data

import argparse

def sortnregress(X, use_lasso):
    """ Take n x d data, order nodes by marginal variance and
    regresses each node onto those with lower variance, using
    edge coefficients as structure estimates. """
    LR = LinearRegression()
    LL = LassoLarsIC(criterion='bic')

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]

        LR.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, covariates] * weight, X[:, target].ravel())
        W[covariates, target] = LL.coef_ * weight
      
        if not use_lasso:
            W[covariates, target] = weight

    return W

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='data file to read in', type=str)
    parser.add_argument('--out', help='file to output w_est to', type=str)
    parser.add_argument('--scale', help='boolean that decides whether to run DOTEARS_scaled', action='store_true')
    parser.add_argument('--use_lasso', action='store_true', help='whether to use lasso penalty')
    args = parser.parse_args()
    print(args.use_lasso)
    # read data
    data = np.load(args.data)
    print(data['obs'].var(axis=0))
    if args.scale:
        data = scale_data(data)
    print(data['obs'].var(axis=0))

    W_est = sortnregress(data['obs'], args.use_lasso)
    
    # W_est[np.abs(W_est) < 0.3] = 0
    np.save(args.out, W_est)
