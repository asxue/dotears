import numpy as np

import argparse
import os
import sys
import copy

def run_dotears(data, lambda1):
    DOTEARS_obj = DOTEARS(data, lambda1=lambda1, scaled=False, w_threshold=0)
    return DOTEARS_obj.fit()

def run_notears(data, lambda1, w_threshold):
    return notears_linear(data['obs'], lambda1=lambda1, w_threshold=w_threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to input data file')
    parser.add_argument('--method', type=str, help='method to run', required=True)
    parser.add_argument('--out', type=str, help='path for output')
    args = parser.parse_args()

    data = dict(np.load(args.data))
    if args.method == 'DOTEARS':
        from dotears import DOTEARS
        w = run_dotears(data, lambda1=0)
    elif args.method == 'dotears_no_omega':
        from dotears import DOTEARS
        DOTEARS_obj = DOTEARS(data, lambda1=0, scaled=False, w_threshold=0)
        DOTEARS_obj.V_inverse = np.identity(DOTEARS_obj.p)
        w = DOTEARS_obj.fit()
    elif args.method == 'NOTEARS':
        from notears import notears_linear
        w = run_notears(data, lambda1=0, w_threshold=0)
    elif args.method == 'sortnregress':
        from sortnregress import sortnregress
        w = sortnregress(data['obs'], use_lasso=False)
    elif args.method == 'GOLEM-EV':
        sys.path.append('./workflow/scripts/golem/src')
        sys.path.append('./workflow/scripts/golem/src/models')
        import golem
        w = golem.golem(copy.deepcopy(data['obs']), lambda_1=0, lambda_2=5.0,
            equal_variances=True, seed=np.random.randint(0, 2**32-1))
    elif args.method == 'GOLEM-NV':
        sys.path.append('./workflow/scripts/golem/src')
        sys.path.append('./workflow/scripts/golem/src/models')
        import golem
        print(golem)
        w_init = golem.golem(copy.deepcopy(data['obs']), lambda_1=0, lambda_2=5.0,
            equal_variances=True, seed=np.random.randint(0, 2**32-1))
        w = golem.golem(copy.deepcopy(data['obs']), lambda_1=0, lambda_2=5.0,
            equal_variances=False, seed=np.random.randint(0, 2**32-1), B_init=w_init)
        print(w)
    elif args.method == 'direct-lingam':
        import lingam
        model = lingam.DirectLiNGAM()
        model.fit(data['obs'])
        w = model.adjacency_matrix_

    dirname = os.path.dirname(args.out)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save(args.out, w)






