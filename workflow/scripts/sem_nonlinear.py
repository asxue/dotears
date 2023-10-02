import numpy as np
from scipy.stats import norm

import argparse

def quadratic_sem_two_nodes(X, c1, c2):
    X = X.copy()

    X[1, :] = c_1 * X[0, :] + c_2 * X[0, :] ** 2 + X[1, :]
    return X

def gen_all_data_kos(mu, var, n, a, c1, c2):
    p = 2

    data = {}
    data['obs'] = quadratic_sem_two_nodes(norm.rvs(loc=mu, scale=np.sqrt(var), size=(p, n)),
                                          c1, c2).T

    X_1 = norm.rvs(loc=mu, scale=np.sqrt(var), size=(p, n))
    X_1[0, :] *= 1 / a
    data['1'] = quadratic_sem_two_nodes(X_1, c1, c2).T

    X_1 = norm.rvs(loc=mu, scale=np.sqrt(var), size=(p, n))
    X_1[1, :] *= 1 / a
    data['1'] = quadratic_sem_two_nodes(X_1, 0, 0).T

    return data


        