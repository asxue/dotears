import numpy as np
import argparse
import os
import itertools
import causaldag
import igraph
import random

def erdos_renyi_dag(p, d):
    dag = causaldag.rand.directed_erdos(p, density=d)
    adj = dag.to_amat()[0][np.ix_(dag.topological_sort(), dag.topological_sort())].astype(np.float64)
    return adj

def scale_free_dag(p, k):
    g = igraph.Graph.Barabasi(n=p, m=k, directed=True)
    return np.asarray(g.get_adjacency().data).T

def get_weighted_dag_from_binary(adj, edge_range):
    p = adj.shape[0]
    for i in range(p):
        for j in range(p):
            if adj[i, j] != 0:
                w = np.random.uniform(*edge_range)
                if np.random.uniform() < 0.5:
                    w = -w
                adj[i, j] = w

    return adj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, help='number of features/size of adjacency matrix')
    parser.add_argument('--d', type=float, help='proportion of edges (Erdos Renyi)')
    parser.add_argument('--k', type=int, help='number of outgoing edges per node (Scale Free)')
    parser.add_argument('--out', type=str, help='out file')
    parser.add_argument('--lower_edge_range', type=float, help='lower bound of uniform range')
    parser.add_argument('--upper_edge_range', type=float, help='upper bound of uniform range')
    parser.add_argument('--n_sims', type=int, help='number of dags to generate')

    args = parser.parse_args()

    edge_range = [args.lower_edge_range, args.upper_edge_range]
    for i in range(args.n_sims):
        np.random.seed(i)
        dag = erdos_renyi_dag(args.p, args.d)
        dag = get_weighted_dag_from_binary(dag, edge_range)
        
        # write dag
        if not os.path.exists(os.path.join(args.out, 'erdos_renyi')):
            os.makedirs(os.path.join(args.out, 'erdos_renyi'))
        np.savetxt(os.path.join(args.out, 'erdos_renyi', 'sim_' + str(i) + '.txt'), dag)

        random.seed(i)
        dag = scale_free_dag(args.p, args.k).astype(np.float64)
        dag = get_weighted_dag_from_binary(dag, edge_range)
        if not os.path.exists(os.path.join(args.out, 'scale_free')):
            os.makedirs(os.path.join(args.out, 'scale_free'))

        np.savetxt(os.path.join(args.out, 'scale_free', 'sim_' + str(i) + '.txt'), dag)
        
