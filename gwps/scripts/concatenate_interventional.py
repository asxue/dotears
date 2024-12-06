import argparse
import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='file of simulation data to run', type=str)
    parser.add_argument('--out', help='file to output inferred dag', type=str)

    args = parser.parse_args()
    data = np.load(args.data)

    X = data['obs']
    print(X.shape)
    for k, v in data.items():
        if k == 'obs':
            continue
        
        X = np.concatenate([X, v])

    np.savez(args.out, obs=X)