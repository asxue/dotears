rule regenerate_dcdi_sims:
    output:
        expand('data/dcdi_sims/data/{experiment}/intervention{sim}.csv', experiment=dcdi_experiments, sim=range(1, 11)),
        expand('data/dcdi_sims/data/{experiment}/DAG{sim}.npy', experiment=dcdi_experiments, sim=range(1, 11)),
        expand('data/dcdi_sims/data/{experiment}/data_interv{sim}.npy', experiment=dcdi_experiments, sim=range(1, 11)),
        expand('data/dcdi_sims/data/{experiment}/data{sim}.npy', experiment=dcdi_experiments, sim=range(1, 11)),
    envmodules:
        'R/4.1.0-BIO'
    conda:
        '../../workflow/envs/dcdi.yml'
    shell:
        """
        mkdir -p data/dcdi_sims/data
        cd data/dcdi_sims/data
        # perfect
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism linear --intervention-type structural \
            --nb-nodes 10 --expected-degree 1 --nb-dag 10 --nb-points 10000 --suffix linear_perfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism linear --intervention-type structural \
            --nb-nodes 10 --expected-degree 4 --nb-dag 10 --nb-points 10000 --suffix linear_perfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism nn_add --intervention-type structural \
            --nb-nodes 10 --expected-degree 1 --nb-dag 10 --nb-points 10000 --suffix nnadd_perfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism nn_add --intervention-type structural \
            --nb-nodes 10 --expected-degree 4 --nb-dag 10 --nb-points 10000 --suffix nnadd_perfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism nn --intervention-type structural \
            --nb-nodes 10 --expected-degree 1 --nb-dag 10 --nb-points 10000 --suffix nn_perfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism nn --intervention-type structural \
            --nb-nodes 10 --expected-degree 4 --nb-dag 10 --nb-points 10000 --suffix nn_perfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        # imperfect 
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism linear --intervention-type parametric \
            --nb-nodes 10 --expected-degree 1 --nb-dag 10 --nb-points 10000 --suffix linear_imperfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism linear --intervention-type parametric \
            --nb-nodes 10 --expected-degree 4 --nb-dag 10 --nb-points 10000 --suffix linear_imperfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism nn_add --intervention-type parametric \
            --nb-nodes 10 --expected-degree 1 --nb-dag 10 --nb-points 10000 --suffix nnadd_imperfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism nn_add --intervention-type parametric \
            --nb-nodes 10 --expected-degree 4 --nb-dag 10 --nb-points 10000 --suffix nnadd_imperfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism nn --intervention-type parametric \
            --nb-nodes 10 --expected-degree 1 --nb-dag 10 --nb-points 10000 --suffix nn_imperfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        python ../../../workflow/scripts/dcdi/data/generation/generate_data.py --mechanism nn --intervention-type parametric \
            --nb-nodes 10 --expected-degree 4 --nb-dag 10 --nb-points 10000 --suffix nn_imperfect --intervention --nb-interventions 10 \
            --obs-data --min-nb-target 1 --max-nb-target 1 --cover
        """


rule rewrite_dcdi_data_dcdi_sims: # write data into npz format.  convert 0.8 and 0.2 train/test into fold level data
    input:
        interventions='data/dcdi_sims/data/{prefix}/intervention{sim}.csv',
        dag='data/dcdi_sims/data/{prefix}/DAG{sim}.npy',
        data='data/dcdi_sims/data/{prefix}/data_interv{sim}.npy',
        data_obs='data/dcdi_sims/data/{prefix}/data{sim}.npy',
    output:
        dag_out='data/dcdi_sims/dags/{prefix}/sim_{sim}.txt',
        data_out='data/dcdi_sims/data/{prefix}/interventional/sim_{sim}.npz',
        cv_out='data/dcdi_sims/data/{prefix}/interventional/cv/sim_{sim}.npz',
        data_obs_out='data/dcdi_sims/data/{prefix}/observational/sim_{sim}.npz',
        cv_obs_out='data/dcdi_sims/data/{prefix}/observational/cv/sim_{sim}.npz',
    conda:
       '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/standardize_dcdi_data.py --interventions {input.interventions} --dag {input.dag} \
                --data {input.data} --dag_out {output.dag_out} --data_out {output.data_out} --cv_out {output.cv_out} \
                --data_obs {input.data_obs} --data_obs_out {output.data_obs_out} --cv_obs_out {output.cv_obs_out}
        """

rule write_cv_folds_dcdi_sims:
    input:
        'data/dcdi_sims/data/{prefix}/{is_interventional}/cv/sim_{sim}.npz'
    output:
        expand('data/dcdi_sims/data/{{prefix}}/{{is_interventional}}/cv/sim{{sim}}/fold{fold}_train.npz',
                    fold=range(std_config['n_folds'])),
        expand('data/dcdi_sims/data/{{prefix}}/{{is_interventional}}/cv/sim{{sim}}/fold{fold}_val.npz', 
                    fold=range(std_config['n_folds'])),
    params:
        out_dir='data/dcdi_sims/data/{prefix}/{is_interventional}/cv/sim{sim}',
        n_folds=std_config['n_folds']
    conda:
        '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/convert_data_to_cv_format.py --data {input} --out_folder {params.out_dir} \
        --folds {params.n_folds} --random_state {wildcards.sim} --sim {wildcards.sim}
        """

rule convert_data_format_dcdi_dcdi_sims:
    input:
        expand('data/dcdi_sims/data/{{prefix}}/interventional/sim_{sim}.npz', 
               sim=range(1, 11))
    output:
        expand('data/dcdi_sims/temp/dcdi/{{prefix}}/interventional/data_interv{sim}.npy', 
               sim=range(1, 11)),
        expand('data/dcdi_sims/temp/dcdi/{{prefix}}/interventional/regime{sim}.csv', 
               sim=range(1, 11)),
        expand('data/dcdi_sims/temp/dcdi/{{prefix}}/interventional/intervention{sim}.csv', 
               sim=range(1, 11)),
        expand('data/dcdi_sims/temp/dcdi/{{prefix}}/interventional/DAG{sim}.npy', 
               sim=range(1, 11)),
        expand('data/dcdi_sims/temp/dcdi/{{prefix}}/interventional/CPDAG{sim}.npy', 
               sim=range(1, 11))
    params:
        in_dir='data/dcdi_sims/data/{prefix}/interventional/',
        out_dir='data/dcdi_sims/temp/dcdi/{prefix}/interventional/',
        # dag_dir=lambda wildcards: join('data/dcdi_sims/dags/', wildcards.prefix.split('/')[0]),
        dag_dir='data/dcdi_sims/dags/{prefix}',
    conda:
        '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/convert_data_to_dcdi_format.py --data {params.in_dir} --out {params.out_dir} \
        --dag {params.dag_dir} --dcdi
        """

rule write_cv_dcdi_dcdi_sims:
    input:
        data='data/dcdi_sims/data/{prefix}/interventional/cv/sim{sim}/fold{fold}_train.npz',
        dag='data/dcdi_sims/dags/{prefix}/sim_{sim}.txt' # probably have to change later
    output:
        data_interv='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}/data_interv{fold}.npy', 
        regime='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}/regime{fold}.csv', 
        intervention='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}/intervention{fold}.csv', 
        dag='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}/DAG{fold}.npy', 
        cpdag='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}/CPDAG{fold}.npy', 
    params:
        out_dir='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}'
    conda:
        '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/dcdi_cv_data_convert.py --data {input.data} --dag_input {input.dag} \
            --data_interv {output.data_interv} --regime {output.regime} --intervention {output.intervention} \
            --dag_output {output.dag} --cpdag {output.cpdag} --out_dir {params.out_dir}
        """

rule dcdi_cv_dcdi_sims:
    input:
        data_interv='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}/data_interv{fold}.npy', 
        regime='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}/regime{fold}.csv', 
        intervention='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}/intervention{fold}.csv', 
        dag='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}/DAG{fold}.npy', 
        cpdag='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}/CPDAG{fold}.npy', 
    output:
        'data/dcdi_sims/out/{prefix}/dcdi-g/cv/sim{sim}/lambda{lambda1}/nlls_{fold}.pkl'
    params:
        in_dir='data/dcdi_sims/temp/dcdi/{prefix}/cv/sim{sim}',
        out_dir='data/dcdi_sims/out/{prefix}/dcdi-g/cv/sim{sim}/lambda{lambda1}/fold{fold}',
        out_nll='data/dcdi_sims/out/{prefix}/dcdi-g/cv/sim{sim}/lambda{lambda1}/fold{fold}/train/nlls.pkl',
        # p=std_config['p'],
        intervention_type=lambda wildcards: 'imperfect' if ('imperfect' in wildcards.prefix) else 'perfect',
        num_layers=lambda wildcards: 2 if (('_nn_' in wildcards.prefix) or ('_nn_add_' in wildcards.prefix) or ('_nnadd_' in wildcards.prefix)) else 0
    envmodules:
        'R/4.1.0-BIO'
    conda:
        '../../workflow/envs/dcdi.yml'
    benchmark:
        'data/dcdi_sims/benchmarks/dcdi-g/{prefix}/sim{sim}_lambda{lambda1}_fold{fold}.benchmark.txt'
    shell:
        """
        mkdir -p {params.out_dir} 
        python workflow/scripts/dcdi/main.py --train --data-path {params.in_dir} --num-vars 10 --i-dataset {wildcards.fold} \
        --exp-path {params.out_dir} --model DCDI-G --intervention --intervention-type {params.intervention_type} \
        --intervention-knowledge known --reg-coeff {wildcards.lambda1} --num-layers 0 --normalize-data
        mv {params.out_nll} {output}
        rm -r {params.out_dir}
        """

rule dcdi_dcdi_sims:
    input:
        'data/dcdi_sims/temp/dcdi/{prefix}/interventional/data_interv{sim}.npy',
        'data/dcdi_sims/temp/dcdi/{prefix}/interventional/regime{sim}.csv',
        'data/dcdi_sims/temp/dcdi/{prefix}/interventional/intervention{sim}.csv',
        'data/dcdi_sims/temp/dcdi/{prefix}/interventional/DAG{sim}.npy',
        'data/dcdi_sims/temp/dcdi/{prefix}/interventional/CPDAG{sim}.npy',
        expand('data/dcdi_sims/out/{{prefix}}/dcdi-g/cv/sim{{sim}}/lambda{lambda1}/nlls_{fold}.pkl',
            lambda1=LAMBDAS, fold=range(std_config['n_folds']))
    output:
        'data/dcdi_sims/out/{prefix}/dcdi-g/sim_{sim}.npy'
    params:
        in_dir='data/dcdi_sims/temp/dcdi/{prefix}/interventional/',
        nll_dir='data/dcdi_sims/out/{prefix}/dcdi-g/cv/sim{sim}',
        out_dir='data/dcdi_sims/out/{prefix}/dcdi-g/sim{sim}',
        out_dag='data/dcdi_sims/out/{prefix}/dcdi-g/sim{sim}/train/DAG.npy',
        # p=std_config['p'],
        intervention_type=lambda wildcards: 'imperfect' if ('imperfect' in wildcards.prefix) else 'perfect',
        num_layers=lambda wildcards: 2 if (('_nn_' in wildcards.prefix) or ('_nn_add_' in wildcards.prefix) or ('_nnadd_' in wildcards.prefix)) else 0,
        lambda_file=config['lambda_file'],
        n_folds=std_config['n_folds']
    conda:
        '../../workflow/envs/dcdi.yml'
    envmodules:
        'R/4.1.0-BIO'
    benchmark:
        'data/dcdi_sims/benchmarks/dcdi-g/{prefix}/sim{sim}.benchmark.txt'
    shell:
        """
        mkdir -p {params.out_dir}
        python workflow/scripts/dcdi_cv.py --lambda_file {params.lambda_file} --nll_dir {params.nll_dir} \
                --n_folds {params.n_folds} --in_dir {params.in_dir} --p 10 --sim {wildcards.sim} \
                --out_dir {params.out_dir} --intervention_type {params.intervention_type} --num_layers 0
        mv {params.out_dag} {output}
        rm -r {params.out_dir}
        """

rule dotears_dcdi_sims:
    input:
        data='data/dcdi_sims/data/{prefix}/interventional/sim_{sim}.npz',
        cv_train=expand('data/dcdi_sims/data/{{prefix}}/interventional/cv/sim{{sim}}/fold{fold}_train.npz', 
                        fold=range(std_config['n_folds'])),
        cv_val=expand('data/dcdi_sims/data/{{prefix}}/interventional/cv/sim{{sim}}/fold{fold}_val.npz',
                      fold=range(std_config['n_folds'])),
        lambda1=config['lambda_file'],
    output:
        'data/dcdi_sims/out/{prefix}/dotears/sim_{sim}.npy'
    conda:
        '../../workflow/envs/dotears.yml'
    params:
        grid_out='data/dcdi_sims/data/param_grid/dotears/{prefix}/sim_{sim}.csv',
        cv_folder='data/dcdi_sims/data/{prefix}/interventional/cv/sim{sim}',
        n_folds=std_config['n_folds']
    benchmark:
        'data/dcdi_sims/benchmarks/dotears/{prefix}/sim{sim}.benchmark.txt'
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} \
            --data {input.data} --out {output} --lambdas {input.lambda1} --method dotears \
            --param_out {params.grid_out} --folds {params.n_folds}
        """

rule sortnregress_dcdi_sims:
    input:
        'data/dcdi_sims/data/{prefix}/interventional/sim_{sim}.npz'
    output:
        'data/dcdi_sims/out/{prefix}/sortnregress/sim_{sim}.npy'
    conda:
        '../../workflow/envs/dotears.yml'
    benchmark:
        'data/dcdi_sims/benchmarks/sortnregress/{prefix}/sim{sim}.benchmark.txt'
    shell:
        """
        python workflow/scripts/sortnregress.py --data {input}  --out {output} --use_lasso
        """

rule golem_ev_dcdi_sims:
    input:
        data='data/dcdi_sims/data/{prefix}/observational/sim_{sim}.npz',
        lambda1=config['lambda_file'],
        lambda2=std_config['golem_lambda2_file'],
        cv_train=expand('data/dcdi_sims/data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_train.npz',
                        fold=range(std_config['n_folds'])),
        cv_val=expand('data/dcdi_sims/data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_val.npz', fold=range(std_config['n_folds'])),
    output:
        'data/dcdi_sims/out/{prefix}/golem-ev/sim_{sim}.npy'
    conda:
        '../../workflow/envs/golem.yml'
    benchmark:
        'data/dcdi_sims/benchmarks/golem-ev/{prefix}/sim{sim}.benchmark.txt'
    params:
        grid_out='data/dcdi_sims/data/param_grid/golem-ev/{prefix}/sim_{sim}.csv',
        cv_folder='data/dcdi_sims/data/{prefix}/observational/cv/sim{sim}',
        n_folds=std_config['n_folds']
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} --out {output} \
        --lambdas {input.lambda1} --lambda2s {input.lambda2} --method golem-ev --param_out {params.grid_out} \
        --folds {params.n_folds}
        """

rule golem_nv_dcdi_sims:
    input:
        data='data/dcdi_sims/data/{prefix}/observational/sim_{sim}.npz',
        lambda1=config['lambda_file'],
        lambda2=std_config['golem_lambda2_file'],
        cv_train=expand('data/dcdi_sims/data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_train.npz',
                        fold=range(std_config['n_folds'])),
        cv_val=expand('data/dcdi_sims/data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_val.npz', fold=range(std_config['n_folds'])),
    output:
        'data/dcdi_sims/out/{prefix}/golem-nv/sim_{sim}.npy'
    conda:
        '../../workflow/envs/golem.yml'
    benchmark:
        'data/dcdi_sims/benchmarks/golem-nv/{prefix}/sim{sim}.benchmark.txt'
    params:
        grid_out='data/dcdi_sims/data/param_grid/golem-nv/{prefix}/sim_{sim}.csv',
        cv_folder='data/dcdi_sims/data/{prefix}/observational/cv/sim{sim}',
        n_folds=std_config['n_folds']
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} \
        --out {output} --lambdas {input.lambda1} --lambda2s {input.lambda2} --method golem-nv --param_out {params.grid_out} \
        --folds {params.n_folds}
        """

rule direct_lingam_dcdi_sims:
    input:
        'data/dcdi_sims/data/{prefix}/observational/sim_{sim}.npz'
    output:
        'data/dcdi_sims/out/{prefix}/direct-lingam/sim_{sim}.npy'
    conda:
        '../../workflow/envs/lingam.yml'
    benchmark:
        'data/dcdi_sims/benchmarks/direct-lingam/{prefix}/sim{sim}.benchmark.txt'
    shell:
        """
        python workflow/scripts/run_naive.py --data {input} --method direct-lingam --out {output}
        """

rule igsp_dcdi_sims:
    input:
        'data/dcdi_sims/data/{prefix}/interventional/sim_{sim}.npz'
    output:
        'data/dcdi_sims/out/{prefix}/igsp/sim_{sim}.npy'
    conda:
        '../../workflow/envs/dotears.yml'
    params:
        alpha=config['alpha'],
        alpha_inv=config['alpha_inv']
    benchmark:
        'data/dcdi_sims/benchmarks/igsp/{prefix}/sim{sim}.benchmark.txt'
    shell:
        """
        python workflow/scripts/igsp.py --data {input}  --out {output} --alpha {params.alpha} --alpha_inv {params.alpha_inv}
        """

rule ut_igsp_dcdi_sims:
    input:
        'data/dcdi_sims/data/{prefix}/interventional/sim_{sim}.npz'
    output:
        'data/dcdi_sims/out/{prefix}/ut-igsp/sim_{sim}.npy'
    conda:
        '../../workflow/envs/dotears.yml'
    params:
        alpha=config['alpha'],
        alpha_inv=config['alpha_inv']
    benchmark:
        'data/dcdi_sims/benchmarks/ut-igsp/{prefix}/sim{sim}.benchmark.txt'
    shell:
        """
        python workflow/scripts/ut_igsp.py --data {input}  --out {output} --alpha {params.alpha} --alpha_inv {params.alpha_inv}
        """

rule gies_dcdi_sims:
    input:
        'data/dcdi_sims/data/{prefix}/interventional/sim_{sim}.npz'
    output:
        'data/dcdi_sims/out/{prefix}/gies/sim_{sim}.npy'
    conda:
        '../../workflow/envs/gies.yml'
    benchmark:
        'data/dcdi_sims/benchmarks/gies/{prefix}/sim{sim}.benchmark.txt'
    shell:
        """
        Rscript workflow/scripts/gies.R --data {input} --out {output}
        """

rule notears_dcdi_sims:
    input:
        data='data/dcdi_sims/data/{prefix}/observational/sim_{sim}.npz',
        cv_train=expand('data/dcdi_sims/data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_train.npz',
                        fold=range(std_config['n_folds'])),
        cv_val=expand('data/dcdi_sims/data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_val.npz',
                      fold=range(std_config['n_folds'])),
        lambda1=config['lambda_file'],
    output:
        'data/dcdi_sims/out/{prefix}/notears/sim_{sim}.npy'
    conda:
        '../../workflow/envs/dotears.yml'
    params:
        grid_out='data/dcdi_sims/data/param_grid/notears/{prefix}/sim_{sim}.csv',
        cv_folder='data/dcdi_sims/data/{prefix}/observational/cv/sim{sim}',
        n_folds=std_config['n_folds']
    benchmark:
        'data/dcdi_sims/benchmarks/notears/{prefix}/sim{sim}.benchmark.txt'
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} \
        --out {output} --lambdas {input.lambda1} --method notears --param_out {params.grid_out} \
        --folds {params.n_folds}
        """
