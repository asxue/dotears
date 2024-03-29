# dotears:Scalable, consistent DAG estimation using observational and interventional data

This repository is the official implementation of [dotears:Scalable, consistent DAG estimation using observational and interventional data](https://arxiv.org/abs/2305.19215). 

## Requirements
This repository uses snakemake for reproducible workflows. For installation, see the [snakemake installation instructions](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html).

Environment .yml files are stored in ./workflow/envs. Each method is run with std.yml, unless they are given a separate (named) .yml file in ./workflow/envs, i.e. dcdi.yml.

## Experiments
To run the experiments in the paper, run this command:

```experiment
snakemake
```

A configuration for cluster engines is given in config/cluster.json. For execution on your specific cluster, see snakemake's guidelines on cluster execution.

Experimental data is contained in ./data/. Data for two and three node simulations is contained in ./data/small, while data for large random graphs is in the folders ./data/p50\*. 

## Real Data
Real data experiments have been added in ./gwps. Some analyses may require external files from either [stringdb](https://string-db.org/) and/or the [Genome-Wide Perturb-Seq](https://gwps.wi.mit.edu/)

## Configuration
The folder ./config gives the global configuration file ./config/config.yml, which includes the path to the lambda grid for cross validation as well as alpha and alphainv. alpha and alphainv are parameters for running UT-IGSP and IGSP, and does not have any relation to $\alpha$ in the paper.

The folder ./config/simulation\_configs gives simulation parameter details for both small (2, 3 node DAG) simulations in ./config/simulation\_configs/small\_sims\_config.yml, as well as large random simulations in ./config/simulation\_configs/standard\_sims\_config.yml. In these files, 'a' represents $\alpha$ in the paper. 

## Data Format
Simulated data are generated by workflow/scripts/sem.py, and come as .npz files. These are dictionary-like objects, where the key is a string mapped to a $n_k$ by $p$ matrix, and represents an intervention on the node int(key). The special key 'obs' denotes observational matrices, while interventions are 0-indexed, i.e. 0 to $p-1$. Inferred adjacency matrices are output as .npy files.

Data to run DCDI-G is formatted differently (see workflow/scripts/convert\_data\_to\_dcdi\_format.py)

## Training

For convenience, the standard dotears script has been provided at ./dotears.py. To train dotears, run this command (see workflow/rules/standard.smk for cross-validation run on separate data instance)

```train
python dotears.py --data {path-to-npz-data-file} --lambda1 {regularization-parameter} --out {path-to-npy-output}
```


