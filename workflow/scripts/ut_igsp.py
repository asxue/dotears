import numpy as np
from causaldag import unknown_target_igsp
import causaldag as cd
from conditional_independence import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester
from conditional_independence import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester

import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='data file to read in')
    parser.add_argument('--out', type=str, help='file to output w_est to')
    parser.add_argument('--alpha', type=float, help='significance cutoff for conditional independence')
    parser.add_argument('--alpha_inv', type=float, help='significance cutoff for invariance test')
    args = parser.parse_args()
    
    data_dict = np.load(args.data)

    # write into igsp format
    # obs_samples is an ndarray n x p
    obs_data = data_dict['obs']

    p = obs_data.shape[1]
    nodes = list(range(p))

    # iv_samples_list is a list of n x p ndarrays

    inv_samples_from_data = [v for k, v in data_dict.items() if k != 'obs']

    # setting_list is list of dicts
    # each dict is key 'intervention' to a list of nodes
    settings_from_data = [dict(known_interventions=[k]) for k, v in data_dict.items() if k != 'obs']

    obs_suffstat = partial_correlation_suffstat(obs_data)
    invariance_suffstat = gauss_invariance_suffstat(obs_data, inv_samples_from_data)

    alpha = args.alpha
    alpha_inv = args.alpha_inv

    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

    est_dag, est_targets_list  = unknown_target_igsp(settings_from_data, nodes, ci_tester, invariance_tester)

    W_est = est_dag.to_amat()[0]

    np.save(args.out, W_est)
