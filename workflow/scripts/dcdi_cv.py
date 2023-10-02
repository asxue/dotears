import pandas
import numpy as np
import os
import pandas as pd

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_file', type=str, help='path to file with lambdas')
    parser.add_argument('--nll_dir', type=str, help='path to directory with lambda/fold nlls')
    parser.add_argument('--n_folds', type=int, help='number of cv folds')
    parser.add_argument('--in_dir', type=str, help='path to input data directory')
    parser.add_argument('--p', type=int, help='number of variables')
    parser.add_argument('--sim', type=int, help='simulation number')
    parser.add_argument('--out_dir', type=str, help='path to output directory')
    parser.add_argument('--intervention_type', type=str, help='perfect or imperfect intervention')
    parser.add_argument('--num_layers', type=int, help='number of hidden lyaers')
    args = parser.parse_args()

    df = pd.DataFrame(columns=['lambda', 'loss'])

    LAMBDAS = [str(x) for x in list(np.loadtxt(args.lambda_file))]

    for lambda1 in LAMBDAS:
        total_loss = 0

        for fold in range(args.n_folds):
            nll_losses = np.load(os.path.join(args.nll_dir,
                                      f'lambda{lambda1}/nlls_{fold}.pkl'),
                                      allow_pickle=True)
            total_loss += nll_losses[-1]

        df.loc[len(df.index)] = [lambda1, total_loss / args.n_folds]

    chosen_lambda = df.loc[df['loss'] == df['loss'].min(), 'lambda'].values[0]

    os.system(f'python workflow/scripts/dcdi/main.py --train --data-path {args.in_dir} --num-vars {args.p} \
                --i-dataset {args.sim} --exp-path {args.out_dir} --model DCDI-G --intervention \
                --intervention-type {args.intervention_type} \
                --intervention-knowledge known --reg-coeff {chosen_lambda} --num-layers {args.num_layers} \
                --normalize-data')
    