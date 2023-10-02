import numpy as np
import argparse
import os
from scipy.stats import norm

from sem import gen_random_variances, split_interventional_data_cv

def fixed_intervention_sem(b_t, mu, var, n, target, mu_intervention, var_intervention):
    p = b_t.shape[0]
    X = norm.rvs(loc=mu, scale=np.sqrt(var), size=(p, n))
    
    for i in range(p):
        if target == str(i):
            X[i, :] = norm.rvs(loc=mu_intervention, scale=np.sqrt(var_intervention), size=(1, n))
        else:
            X[i, :] = np.matmul(b_t[i, :], X)
        
    return X

def gen_fixed_intervention_data(b_t, mu, var, n, parental_influence=0, a_perturbation=False, mu_intervention=2, var_intervention=1):
    p = b_t.shape[0]
    
    data = {}
    data['obs'] = fixed_intervention_sem(b_t, mu, var, n, 'obs', mu_intervention, var_intervention).T

    for i in range(p):
        data[str(i)] = fixed_intervention_sem(b_t, mu, var, n, str(i), mu_intervention, var_intervention).T
        
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, help='output file for data')
    parser.add_argument('--b_template', type=str, help='input file for b_template adjacency matrix')
    parser.add_argument('--var', type=str, help='variances file for nodes')
    parser.add_argument('--random_var', action='store_true', help='draw random variances')
    parser.add_argument('--std_lower_range', type=float, help='lower bound of std')
    parser.add_argument('--std_upper_range', type=float, help='upper bound of std')
    parser.add_argument('--n', type=int)
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

    data = gen_fixed_intervention_data(b_t, np.zeros((p, 1)), var=sim_var, n=n,
                             parental_influence=args.parental_influence)

    if args.a_perturbation:
        data = gen_fixed_intervention_data(b_t, np.zeros((p, 1)), var=sim_var, n=n,  
            parental_influence=args.parental_influence, a_perturbation=True)

    if is_observational:
        np.savez(args.out, obs=data['obs'])
    # save data 
    else:
        np.savez(args.out, **data)