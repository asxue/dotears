import numpy as np
import pandas as pd
import os
import argparse

from sklearn.model_selection import train_test_split

def read_intervention_file(path):
    interventions = np.asarray([]).astype(str)
    
    with open(path, 'r') as f:
        for line in f:
            if line == '\n':
                intervention = 'obs'
            else:
                intervention = int(line)

            interventions = np.append(interventions, intervention)
            
    return interventions
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interventions', type=str, help='path to intervention help')
    parser.add_argument('--dag', type=str, help='path to dag')
    parser.add_argument('--data', type=str, help='path to data')
    parser.add_argument('--data_out', type=str, help='path to data output file')
    parser.add_argument('--cv_out', type=str, help='path to cv output file')
    parser.add_argument('--dag_out', type=str, help='path to dag output file' )
    parser.add_argument('--data_obs', type=str, help='path to obs data output file')
    parser.add_argument('--data_obs_out', type=str, help='path to observational data out')
    parser.add_argument('--cv_obs_out', type=str, help='path to observational cv data output file')
    args = parser.parse_args()

    interventions = read_intervention_file(args.interventions)
    data = np.load(args.data)
    dag = np.load(args.dag)

    out = {}
    cv_out = {}

    for intervention in np.unique(interventions):
        data_k = data[interventions == intervention, :]

        cv_data, out_data = train_test_split(data_k, test_size=0.2)
        cv_out[intervention] = cv_data
        out[intervention] = out_data

    data_obs = np.load(args.data_obs)
    cv_data_obs, out_data_obs = train_test_split(data_obs, test_size=0.2)

    obs_data_out = {'obs': out_data_obs}
    obs_cv_out = {'obs': cv_data_obs}
    np.savetxt(args.dag_out, dag)
    np.savez(args.data_out, **out)
    np.savez(args.cv_out, **cv_out)
    np.savez(args.data_obs_out, **out)
    np.savez(args.cv_obs_out, **out)
