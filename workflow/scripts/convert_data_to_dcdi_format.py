import numpy as np
import os
import argparse
import causaldag

"""
Converts .npz data file into a format recognizable by dcdi. Details at
https://github.com/slachapelle/dcdi
"""

def convert_data_to_dcdi_format(data):
    obs_data = data['obs']
    n_obs, p = obs_data.shape
    
    dcdi_data = obs_data
    dcdi_regime = n_obs * np.zeros(n_obs)
    dcdi_interventions = np.asarray(['' for _ in range(n_obs)])
    
    i = 1
    for k in data.keys():
        if k == 'obs':
            continue
        
        X = data[k]
        n_k = X.shape[0]
        dcdi_data = np.vstack([X, dcdi_data])
        dcdi_regime = np.concatenate([int(i) * np.ones(n_k).astype(int), dcdi_regime])
        dcdi_interventions = np.concatenate([int(k) * np.ones(n_k).astype(int), dcdi_interventions])
        
        i += 1
    print(dcdi_data.shape)
    dcdi_regime = dcdi_regime.astype(int)
    return dcdi_data, dcdi_regime.astype(str), dcdi_interventions.astype(str)

if __name__ == '__main__':
    """
    data_dir: path to .npz file with simulation data
    dag_dir: path to .txt file with DAG adjacency matrix
    out: parent directory for output converted data
    n_sims: iterate over sim numbers 0 to n_sims - 1
    small: ignores the dag_dir param, since there is only one DAG file
    small_dag_path: path to small DAG file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data file to read in', type=str)
    parser.add_argument('--dag_dir', help='dag data file to read in', type=str)
    parser.add_argument('--out', help='directory to output converted data to', type=str)
    parser.add_argument('--n_sims', type=int, help='number of simulations')
    parser.add_argument('--small', action='store_true', help='specify a single dag file in small simulations')
    parser.add_argument('--small_dag_path', type=str, help='path to dag file in small simulations')
    args = parser.parse_args()
    
    for i in range(args.n_sims):
        data_path = os.path.join(args.data_dir, 'sim_{}.npz'.format(i))
        
        if args.small:
            dag_path = args.small_dag_path
        else: 
            dag_path = os.path.join(args.dag_dir, 'sim_{}.txt'.format(i))
  
        data = np.load(data_path)
        dag = np.loadtxt(dag_path)
        cpdag = causaldag.DAG.from_amat(dag).cpdag().to_amat()[0] * 1.

        dcdi_data, regime, interventions = convert_data_to_dcdi_format(data)
        dcdi_data_file = os.path.join(args.out, 'data_interv{}.npy'.format(i))
        regime_file = os.path.join(args.out, 'regime{}.csv'.format(i))
        intervention_file = os.path.join(args.out, 'intervention{}.csv'.format(i))
        dag_file = os.path.join(args.out, 'DAG{}.npy'.format(i))
        cpdag_file = os.path.join(args.out, 'CPDAG{}.npy'.format(i))

        if not os.path.exists(args.out):
            os.makedirs(args.out)

        np.save(dcdi_data_file, dcdi_data)
        np.savetxt(regime_file, regime, fmt='%s')
        np.savetxt(intervention_file, interventions, fmt='%s')
        np.save(dag_file, dag)
        np.save(cpdag_file, cpdag)
