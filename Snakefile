from os.path import join
import yaml
import numpy as np

configfile: 'config/config.yml'

def load_sim_configs(path):
    with open(path, 'r') as f:
        out = yaml.full_load(f)
    return out

small_config_path = 'config/simulation_configs/small_sims_config.yml'
std_config_path = 'config/simulation_configs/standard_sims_config.yml'
small_config = load_sim_configs(small_config_path)

SMALL_SIGMA_1S = [1, 2, 4, 10, 100] # 1, 2, 4, 8, 16...
SMALL_BETAS = [np.round(x, 2) for x in list(np.arange(0.1, 1.6, .1))]
SMALL_DAGS = ['chain_2', 'chain_3', 'collider_3', 'fork_3']

SMALL_METHODS = ['dotears', 'notears', 'sortnregress', 'golem-ev', 'golem-nv', 'direct-lingam', 'gies', 'igsp', 'ut-igsp', 'dotears_obsonly']
np.savetxt(small_config['beta_file'], np.asarray(SMALL_BETAS))
np.savetxt(small_config['sigma_1_file'], np.asarray(SMALL_SIGMA_1S))

std_config = load_sim_configs(std_config_path)
STD_METHODS = SMALL_METHODS

data_types = ['raw']
dag_models = ['erdos_renyi', 'scale_free']
LAMBDAS = [str(x) for x in list(np.loadtxt(config['lambda_file']))]
rule all:
    input:
      expand(join(small_config['output_dir'], 'out/{method}/{dag}/sigma1_{sigma1}/beta_{beta}/sim_{sim}.npy'), method=SMALL_METHODS, dag=SMALL_DAGS, sigma1=SMALL_SIGMA_1S, beta=SMALL_BETAS, sim=range(small_config['n_sims'])),
      expand(join(small_config['output_dir'], 'out/dcdi-g/{dag}/sigma1_{sigma1}/beta_{beta}/sim_{sim}.npy'), dag=SMALL_DAGS, sigma1=SMALL_SIGMA_1S, beta=SMALL_BETAS, sim=range(small_config['n_sims'])),
      expand(join(std_config['output_dir'], 'out/{dag_model}/{data_type}/{method}/sim_{sim}.npy'), dag_model=dag_models, data_type=data_types, method=STD_METHODS, sim=range(std_config['n_sims'])),
#       expand(join(std_config['output_dir'], 'out/{dag_model}/{data_type}/dcdi-g/lambda{lambda1}/sim_{sim}.npy'), dag_model=dag_models, data_type=data_types, lambda1=LAMBDAS, sim=range(std_config['dcdi_n_sims'])),

include: 'workflow/rules/small_graphs.smk'
include: 'workflow/rules/standard.smk'
