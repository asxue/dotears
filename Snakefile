from os.path import join, basename
import yaml
import numpy as np
import glob

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
# SMALL_DAGS = ['chain_2', 'chain_3', 'collider_3', 'fork_3']
SMALL_DAGS = ['chain_2']


METHODS = ['dotears', 'notears', 'sortnregress', 'golem-ev', 'golem-nv', 'direct-lingam', 'gies', 'igsp', 'ut-igsp', 'dcdi-g']
# METHODS = ['dcdi-g']

SMALL_METHODS = METHODS + ['dotears_no_omega']


np.savetxt(small_config['beta_file'], np.asarray(SMALL_BETAS))
np.savetxt(small_config['sigma_1_file'], np.asarray(SMALL_SIGMA_1S))

std_config = load_sim_configs(std_config_path)

dag_models = ['erdos_renyi', 'scale_free']

parental_influences = [0.0, 0.1, 0.25, 0.5]
LAMBDAS = [str(x) for x in list(np.loadtxt(config['lambda_file']))]

experiments = [
    'standard',
#     'a_perturbed',
#     'fixed_intervention',
#     'low_sample_size'
]

dcdi_experiments = [
    'data_p10_e10.0_n10000_linear_perfect',
    'data_p10_e40.0_n10000_linear_perfect',
    'data_p10_e10.0_n10000_nnadd_perfect',
    'data_p10_e40.0_n10000_nnadd_perfect',
    'data_p10_e10.0_n10000_nn_perfect',
    'data_p10_e40.0_n10000_nn_perfect',
    'data_p10_e10.0_n10000_linear_imperfect',
    'data_p10_e40.0_n10000_linear_imperfect',
    'data_p10_e10.0_n10000_nnadd_imperfect',
    'data_p10_e40.0_n10000_nnadd_imperfect',
    'data_p10_e10.0_n10000_nn_imperfect',
    'data_p10_e40.0_n10000_nn_imperfect',
]

rule all:
    input:
        expand(join(std_config['output_dir'], 'out/{experiment}/{dag_model}/{method}/sim_{sim}.npy'),
               dag_model=dag_models, experiment=experiments, method=METHODS, sim=range(std_config['n_sims'])),
        expand(join(small_config['output_dir'], 'out/{method}/{dag}/sigma1_{sigma1}/beta_{beta}/sim_{sim}.npy'), 
               method=SMALL_METHODS, dag=SMALL_DAGS, sigma1=SMALL_SIGMA_1S, 
               beta=SMALL_BETAS, sim=range(small_config['n_sims'])),
        # expand(join(std_config['output_dir'], 'out/pi_{pi}/{dag_model}/{method}/sim_{sim}.npy'),
        #        dag_model=dag_models, pi=parental_influences, method=METHODS, sim=range(std_config['n_sims'])),
#        expand('data/dcdi_sims/out/{experiment}/{method}/sim_{sim}.npy',
#               experiment=dcdi_experiments, method=METHODS, sim=range(1, 11)),

include: 'workflow/rules/generate_data.smk'
include: 'workflow/rules/standard.smk'
include: 'workflow/rules/small_graphs.smk' 
include: 'workflow/rules/dcdi.smk'
include: 'workflow/rules/dcdi_sims.smk'
