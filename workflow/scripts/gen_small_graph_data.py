import argparse
import numpy as np
import os
from sem import gen_all_data_kos

def get_dag_from_path(path):
    f = os.path.basename(path)
    return f.split('.')[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_template', type=str, help='path to file for adjacency matrix')
    parser.add_argument('--sigma_1_file', type=str, help='path to file with sigma_1s')
    parser.add_argument('--beta_file', type=str, help='path to file with betas')
    parser.add_argument('--n', type=int, help='sample size')
    parser.add_argument('--n_sims', type=int, help='number of simulations')
    parser.add_argument('--a', type=float, help='variance reduction upon intervention')
    parser.add_argument('--out', type=str, help='output base directory for data')
    parser.add_argument('--observational_data', help='generate observational data', action='store_true')
    args = parser.parse_args()

    # accept a template file
    w_template = np.loadtxt(args.w_template)
    p = w_template.shape[0]
    name = get_dag_from_path(args.w_template)

    # accept a beta file
    betas = np.loadtxt(args.beta_file).astype(np.float64)

    # accept a sigma_1^2 file; hold sigma_2^2 = 1
    sigmas = np.loadtxt(args.sigma_1_file).astype(np.int64)
    # multiply the template by beta
    for beta in betas:
        w_template_beta = w_template * beta
        w_t = np.transpose(w_template_beta) + np.identity(p)

        for sigma_1 in sigmas:
            if p == 2:
                sim_var = np.asarray([sigma_1, 1]).reshape(p, 1)
            if p == 3:
                sim_var = np.asarray([sigma_1, 1, 1]).reshape(p, 1) 

            for sim in range(args.n_sims):
                np.random.seed(sim) 
                if args.observational_data:
                    data = gen_all_data_kos(w_t, np.zeros((p, 1)), var=sim_var, n=(p + 1) * args.n, a=args.a)
                    parent = os.path.join(args.out, 'observational/{}/sigma1_{}/beta_{}/'.format(name, sigma_1, beta))
                    if not os.path.exists(parent):
                        os.makedirs(parent)
                    out_file = os.path.join(parent, 'sim_{}.npz'.format(sim))
                    np.savez(out_file, obs=data['obs'])
                else:
                    data = gen_all_data_kos(w_t, np.zeros((p, 1)), var=sim_var, n=args.n, a=args.a)
                    parent = os.path.join(args.out, 'interventional/{}/sigma1_{}/beta_{}/'.format(name, sigma_1, beta))
                    if not os.path.exists(parent):
                        os.makedirs(parent)
                    out_file = os.path.join(parent, 'sim_{}.npz'.format(sim))
                    np.savez(out_file, **data)
