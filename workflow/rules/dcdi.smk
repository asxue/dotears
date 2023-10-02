# rule rewrite_dcdi_data:
#     input:
#         interventions='workflow/scripts/dcdi/data/{intervention_type}/{data_type}/intervention{sim}.csv',
#         dag='workflow/scripts/dcdi/data/{intervention_type}/{data_type}/DAG{sim}.npy',
#         data='workflow/scripts/dcdi/data/{intervention_type}/{data_type}/data_interv{sim}.npy'
#     output:
#         dag_out='data/dcdi_sims/dags/{intervention_type}/{data_type}/sim_{sim}.txt',
#         data_out='data/dcdi_sims/data/{intervention_type}/{data_type}/interventional/sim_{sim}.npz',
#     shell:
#         """
#         python workflow/out/standardize_dcdi_data --interventions {input.interventions} --dag {input.dag} \
#                 --data {input.data} --dag_out {output.dag_out} --data_out {output.data_out}
#         """

# ruleorder: rewrite_dcdi_data > generate_standard_data

rule convert_data_format_dcdi:
    input:
        expand(join(std_config['output_dir'], 
               'data/{{prefix}}/interventional/sim_{sim}.npz'), 
               sim=range(std_config['n_sims']))
    output:
        expand(join(std_config['output_dir'], 
               'temp/dcdi/{{prefix}}/interventional/data_interv{sim}.npy'), 
               sim=range(std_config['n_sims'])),
        expand(join(std_config['output_dir'], 
               'temp/dcdi/{{prefix}}/interventional/regime{sim}.csv'), 
               sim=range(std_config['n_sims'])),
        expand(join(std_config['output_dir'], 
               'temp/dcdi/{{prefix}}/interventional/intervention{sim}.csv'), 
               sim=range(std_config['n_sims'])),
        expand(join(std_config['output_dir'], 
               'temp/dcdi/{{prefix}}/interventional/DAG{sim}.npy'), 
               sim=range(std_config['n_sims'])),
        expand(join(std_config['output_dir'], 
               'temp/dcdi/{{prefix}}/interventional/CPDAG{sim}.npy'), 
               sim=range(std_config['n_sims']))
    params:
        in_dir=join(std_config['output_dir'],  'data/{prefix}/interventional/'),
        out_dir=join(std_config['output_dir'], 'temp/dcdi/{prefix}/interventional/'),
        dag_dir=join(std_config['output_dir'], 'dags/{prefix}'),
        n_sims=std_config['n_sims'] 
    conda:
        '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/convert_data_to_dcdi_format.py --data {params.in_dir} --out {params.out_dir} \
        --n_sims {params.n_sims} --dag {params.dag_dir}
        """

rule write_cv_dcdi:
    input:
        data=join(std_config['output_dir'], 'data/{prefix}/interventional/cv/sim{sim}/fold{fold}_train.npz'),
        dag=join(std_config['output_dir'], 'dags/{prefix}', 'sim_{sim}.txt') # probably have to change later
    output:
        data_interv=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}/data_interv{fold}.npy'), 
        regime=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}/regime{fold}.csv'), 
        intervention=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}/intervention{fold}.csv'), 
        dag=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}/DAG{fold}.npy'), 
        cpdag=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}/CPDAG{fold}.npy'), 
    params:
        out_dir=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}')
    conda:
        '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/dcdi_cv_data_convert.py --data {input.data} --dag_input {input.dag} \
            --data_interv {output.data_interv} --regime {output.regime} --intervention {output.intervention} \
            --dag_output {output.dag} --cpdag {output.cpdag} --out_dir {params.out_dir}
        """

rule dcdi_cv:
    input:
        data_interv=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}/data_interv{fold}.npy'), 
        regime=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}/regime{fold}.csv'), 
        intervention=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}/intervention{fold}.csv'), 
        dag=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}/DAG{fold}.npy'), 
        cpdag=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}/CPDAG{fold}.npy'), 
    output:
        join(std_config['output_dir'], 'out/{prefix}/dcdi-g/cv/sim{sim}/lambda{lambda1}/nlls_{fold}.pkl')
    params:
        in_dir=join(std_config['output_dir'], 'temp/dcdi/{prefix}/cv/sim{sim}'),
        out_dir=join(std_config['output_dir'], 'out/{prefix}/dcdi-g/cv/sim{sim}/lambda{lambda1}/fold{fold}'),
        out_nll=join(std_config['output_dir'], 'out/{prefix}/dcdi-g/cv/sim{sim}/lambda{lambda1}/fold{fold}/train/nlls.pkl'),
        p=std_config['p'],
        intervention_type=lambda wildcards: 'imperfect' if (('pi_' in wildcards.prefix) & ('pi_0.0' not in wildcards.prefix)) else 'perfect'
    envmodules:
        'R/4.1.0-BIO'
    conda:
        '../../workflow/envs/dcdi.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/dcdi-g/{prefix}/sim{sim}_lambda{lambda1}_fold{fold}.benchmark.txt')
    shell:
        """
        mkdir -p {params.out_dir} 
        module load R/4.1.0-BIO
        python workflow/scripts/dcdi/main.py --train --data-path {params.in_dir} --num-vars {params.p} --i-dataset {wildcards.fold} \
        --exp-path {params.out_dir} --model DCDI-G --intervention --intervention-type {params.intervention_type} \
        --intervention-knowledge known --reg-coeff {wildcards.lambda1} --num-layers 0 --normalize-data 
        mv {params.out_nll} {output}
        rm -r {params.out_dir}
        """

rule dcdi:
    input:
        join(std_config['output_dir'], 'temp/dcdi/{prefix}/interventional/data_interv{sim}.npy'),
        join(std_config['output_dir'], 'temp/dcdi/{prefix}/interventional/regime{sim}.csv'),
        join(std_config['output_dir'], 'temp/dcdi/{prefix}/interventional/intervention{sim}.csv'),
        join(std_config['output_dir'], 'temp/dcdi/{prefix}/interventional/DAG{sim}.npy'),
        join(std_config['output_dir'], 'temp/dcdi/{prefix}/interventional/CPDAG{sim}.npy'),
        expand(join(std_config['output_dir'], 
                    'out/{{prefix}}/dcdi-g/cv/sim{{sim}}/lambda{lambda1}/nlls_{fold}.pkl'),
                    lambda1=LAMBDAS, fold=range(std_config['n_folds']))
    output:
        join(std_config['output_dir'], 'out/{prefix}/dcdi-g/sim_{sim}.npy')
    params:
        in_dir=join(std_config['output_dir'], 'temp/dcdi/{prefix}/interventional'),
        nll_dir=join(std_config['output_dir'], 'out/{prefix}/dcdi-g/cv/sim{sim}'),
        out_dir=join(std_config['output_dir'], 'out/{prefix}/dcdi-g/out/sim{sim}'),
        out_dag=join(std_config['output_dir'], 'out/{prefix}/dcdi-g/out/sim{sim}/train/DAG.npy'),
        p=std_config['p'],
        intervention_type=lambda wildcards: 'imperfect' if (('pi_' in wildcards.prefix) & ('pi_0.0' not in wildcards.prefix)) else 'perfect',
        lambda_file=config['lambda_file'],
        n_folds=std_config['n_folds']
    conda:
        '../../workflow/envs/dcdi.yml'
    envmodules:
        'R/4.1.0-BIO'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/dcdi-g/{prefix}/sim{sim}.benchmark.txt')
    shell:
        """
        mkdir -p {params.out_dir}
        python workflow/scripts/dcdi_cv.py --lambda_file {params.lambda_file} --nll_dir {params.nll_dir} \
                --n_folds {params.n_folds} --in_dir {params.in_dir} --p {params.p} --sim {wildcards.sim} \
                --out_dir {params.out_dir} --intervention_type {params.intervention_type} --num_layers 0 
        mv {params.out_dag} {output}
        rm -r {params.out_dir}
        """
