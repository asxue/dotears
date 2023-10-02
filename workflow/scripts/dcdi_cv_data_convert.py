from convert_data_to_dcdi_format import convert_data_to_dcdi_format

import numpy as np
import argparse
import os
import causaldag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to npz data file')
    parser.add_argument('--dag_input', type=str, help='path to input dag')
    parser.add_argument('--data_interv', type=str, help='path to output data')
    parser.add_argument('--regime', type=str, help='path to regime file')
    parser.add_argument('--intervention', type=str, help='path to output intervention file')
    parser.add_argument('--dag_output', type=str, help='path to output dag file')
    parser.add_argument('--cpdag', type=str, help='path to output cpdag')
    parser.add_argument('--out_dir', type=str, help='path to output directory')
    args = parser.parse_args()

    data = np.load(args.data)
    dag = np.loadtxt(args.dag_input)

    cpdag = causaldag.DAG.from_amat(dag).cpdag().to_amat()[0] * 1.

    dcdi_data, regime, interventions = convert_data_to_dcdi_format(data)
    dcdi_data_file = os.path.join(args.data_interv)
    regime_file = os.path.join(args.regime)
    intervention_file = os.path.join(args.intervention)
    dag_file = os.path.join(args.dag_output)
    cpdag_file = os.path.join(args.cpdag)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    np.save(dcdi_data_file, dcdi_data)
    np.savetxt(regime_file, regime, fmt='%s')
    np.savetxt(intervention_file, interventions, fmt='%s')
    np.save(dag_file, dag)
    np.save(cpdag_file, cpdag)