import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

import argparse

def sem(X, b_t):
    X = X.copy()
    p_0 = X.shape[0]
    
    for i in range(p_0):
        X[i, :] = np.matmul(b_t[i, :], X)
        
    return X

def latent_normal(b_t, mu, var, n):
    p = b_t.shape[0]
    data = norm.rvs(loc=mu, scale=np.sqrt(var), size=(p, n))
    data = sem(data, b_t)
    
    return data

def ko_struct(b_t, i, a, parental_influence):
    ko_t = np.copy(b_t)
    
    # ko_t[i, :] = np.zeros((1, b_t.shape[0])) 
    ko_t[i, :] = np.ones((1, b_t.shape[0])) * np.sqrt(parental_influence)
    ko_t[i, i] = b_t[i, i] / a
    return ko_t

def gen_all_data_kos(b_t, mu, var, n, a, parental_influence=0, a_perturbation=False):
    p = b_t.shape[0]
    
    data = {}
    data['obs'] = latent_normal(b_t, mu, var, n).T

    for i in range(p):
        if a_perturbation:
            perturbation = np.random.uniform(0.8, 1.2)
            a_i = a * perturbation
        else:
            a_i = a
        b_ko = ko_struct(b_t, i, a_i, parental_influence)

        data[str(i)] = latent_normal(b_ko, mu, var, n).T
        
    return data

def scale_data(data):
    scaler = StandardScaler()
    for k, v in data.items():
        scaled = scaler.fit_transform(v)
        data[k] = scaled
        
    return data

def gen_random_variances(sim_std_range, p):
    sim_std = np.random.uniform(*sim_std_range, size=p)
    return (sim_std ** 2).reshape(p, 1)

def split_interventional_data_cv(data, cv_k):
    train_splits = [({}, {}) for _ in range(cv_k)]
    kf = KFold(n_splits=cv_k, random_state=random_state, shuffle=True)

    for key, arr in data.items():
        for split, (train_cv_index, val_cv_index) in enumerate(kf.split(arr)):
            train_splits[split][0][key] = train_arr[train_cv_index]
            train_splits[split][1][key] = train_arr[val_cv_index]

    return train_splits

if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, help='output file for data')
    parser.add_argument('--b_template', type=str, help='input file for b_template adjacency matrix')
    parser.add_argument('--var', type=str, help='variances file for nodes')
    parser.add_argument('--random_var', action='store_true', help='draw random variances')
    parser.add_argument('--std_lower_range', type=float, help='lower bound of std')
    parser.add_argument('--std_upper_range', type=float, help='upper bound of std')
    parser.add_argument('--n', type=int)
    parser.add_argument('--a', type=float)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--type', type=str, help='generate observational or interventional data')
    parser.add_argument('--cv', action='store_true', help='split data into cross validation folds')
    parser.add_argument('--parental_influence', type=float, default=0)
    parser.add_argument('--a_perturbation', action='store_true', help='perturb the variance of KO')
    args = parser.parse_args()

    np.random.seed(args.seed)
    # read b_template
    b_template = np.loadtxt(args.b_template)
    p = b_template.shape[0]
    b_t = np.transpose(b_template) + np.identity(p)

    # generate data 
    if args.random_var:
        sim_var = gen_random_variances([args.std_lower_range, args.std_upper_range], p)
    else:
        sim_var = np.asarray(list(args.var)).reshape(p, 1)

    is_observational = (args.type == 'observational')
    if is_observational:
        n = (p + 1) * args.n
    else:
        n = args.n

    data = gen_all_data_kos(b_t, np.zeros((p, 1)), var=sim_var, n=n,
                             a=args.a, parental_influence=args.parental_influence)

    if args.a_perturbation:
        data = gen_all_data_kos(b_t, np.zeros((p, 1)), var=sim_var, n=n, a=args.a, 
            parental_influence=args.parental_influence, a_perturbation=True)
        

    if is_observational:
        np.savez(args.out, obs=data['obs'])
    # save data 
    else:
        np.savez(args.out, **data)
