rule generate_dags:
    output:
        er=expand(join(std_config['output_dir'], 'dags/erdos_renyi/sim_{sim}.txt'), sim=range(std_config['n_sims'])),
        sf=expand(join(std_config['output_dir'], 'dags/scale_free/sim_{sim}.txt'), sim=range(std_config['n_sims']))
    params:
        p=std_config['p'],
        d=std_config['d'],
        k=std_config['k'],
        lower_edge_range=std_config['lower_edge_range'],
        upper_edge_range=std_config['upper_edge_range'],
        n_sims=std_config['n_sims'],
        out=join(std_config['output_dir'], 'dags')
    conda:
        '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/erdos_renyi.py --p {params.p} --d {params.d} --k {params.k} --out {params.out} --lower_edge_range {params.lower_edge_range} --upper_edge_range {params.upper_edge_range} --n_sims {params.n_sims}
        """

rule generate_data_from_dag_std:
    input:
        join(std_config['output_dir'], 'dags/{dag_model}/sim_{sim}.txt')
    output:
        out_obs=join(std_config['output_dir'], 'data/{dag_model}/raw/observational/sim_{sim}.npz'),
        out_int=join(std_config['output_dir'], 'data/{dag_model}/raw/interventional/sim_{sim}.npz'),
        out_obs_cv=join(std_config['output_dir'], 'data/{dag_model}/raw/observational/cv/sim_{sim}.npz'),
        out_int_cv=join(std_config['output_dir'], 'data/{dag_model}/raw/interventional/cv/sim_{sim}.npz')
    params:
        n=std_config['n'],
        std_lower_range=std_config['lower_std_range'],
        std_upper_range=std_config['upper_std_range'],
        a=std_config['a'],
        cv_seed='{sim}' + str(std_config['n_sims']) # not beautiful but deterministic
    conda:
       '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/sem.py --out {output.out_obs} --b_template {input} --random_var --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} --a {params.a} --seed {wildcards.sim} --type observational
        python workflow/scripts/sem.py --out {output.out_obs_cv} --b_template {input} --random_var --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} --a {params.a} --seed {params.cv_seed} --type observational
        python workflow/scripts/sem.py --out {output.out_int} --b_template {input} --random_var --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} --a {params.a} --seed {wildcards.sim} --type interventional
        python workflow/scripts/sem.py --out {output.out_int_cv} --b_template {input} --random_var --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} --a {params.a} --seed {params.cv_seed} --type interventional
        """

rule convert_to_scaled_data_std:
    input:
        join(std_config['output_dir'], 'data/{dag_model}/raw/{is_interventional}/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'data/{dag_model}/scaled/{is_interventional}/sim_{sim}.npz')
    conda:
        '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/scale_data.py --data {input} --out {output}
        """    

rule write_cv_folds:
    input:
        join(std_config['output_dir'], 'data/{dag_model}/{data_type}/{is_interventional}/cv/sim_{sim}.npz')    
    output:
        expand(join(std_config['output_dir'], 'data/{{dag_model}}/{{data_type}}/{{is_interventional}}/cv/sim{{sim}}/fold{fold}_train.npz'), fold=range(std_config['n_folds'])),
        expand(join(std_config['output_dir'], 'data/{{dag_model}}/{{data_type}}/{{is_interventional}}/cv/sim{{sim}}/fold{fold}_val.npz'), fold=range(std_config['n_folds'])),
    params:
        out_dir=join(std_config['output_dir'], 'data/{dag_model}/{data_type}/{is_interventional}/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    conda:
        '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/convert_data_to_cv_format.py --data {input} --out_folder {params.out_dir} --folds {params.n_folds} --random_state {wildcards.sim} --sim {wildcards.sim}
        """

rule run_dotears_cv_std_raw:
    input:
        data=join(std_config['output_dir'], 'data/{dag_model}/raw/interventional/sim_{sim}.npz'),
        cv_train=expand(join(std_config['output_dir'], 'data/{{dag_model}}/raw/interventional/cv/sim{{sim}}/fold{fold}_train.npz'), fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'], 'data/{{dag_model}}/raw/interventional/cv/sim{{sim}}/fold{fold}_val.npz'), fold=range(std_config['n_folds'])),
        lambda1=config['lambda_file'],
    output:
        join(std_config['output_dir'], 'out/{dag_model}/raw/dotears/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/dotears/raw/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{dag_model}/raw/interventional/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/dotears/{dag_model}_raw_sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} --out {output} --lambdas {input.lambda1} --method dotears --param_out {params.grid_out} --folds {params.n_folds}
        """

rule run_dotears_obsonly_cv_std_raw:
    input:
        data=join(std_config['output_dir'], 'data/{dag_model}/raw/interventional/sim_{sim}.npz'),
        cv_train=expand(join(std_config['output_dir'], 'data/{{dag_model}}/raw/interventional/cv/sim{{sim}}/fold{fold}_train.npz'), fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'], 'data/{{dag_model}}/raw/interventional/cv/sim{{sim}}/fold{fold}_val.npz'), fold=range(std_config['n_folds'])),
        lambda1=config['lambda_file'],
    output:
        join(std_config['output_dir'], 'out/{dag_model}/raw/dotears_obsonly/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/dotears_obsonly/raw/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{dag_model}/raw/interventional/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/dotears_obsonly/{dag_model}_raw_sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} --out {output} --lambdas {input.lambda1} --method dotears_obsonly --param_out {params.grid_out} --folds {params.n_folds}
        """

# rule run_dotears_cv_std_scaled:
#    input:
#        data=join(std_config['output_dir'], 'data/{dag_model}/raw/interventional/sim_{sim}.npz'), # use raw but scale in object
#        lambda1=config['lambda_file'],
#        w=config['w_file']
#    output:
#        join(std_config['output_dir'], 'out/{dag_model}/scaled/dotears/sim_{sim}.npy')
#    params:
#        grid_out=join(std_config['output_dir'], 'data/param_grid/dotears/scaled/sim_{sim}.csv')
#    conda:
#        '../../workflow/envs/std.yml'
#    shell:
#        """
#        python workflow/scripts/cv.py --data {input.data} --out {output} --lambdas {input.lambda1} --w_file {input.w} --method DOTEARS --scaled --grid_out {params.grid_out}
#        """

rule run_gies_std:
    input:
        join(std_config['output_dir'], 'data/{dag_model}/{is_scaled}/interventional/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/gies/sim_{sim}.npy')
    conda:
        '../../workflow/envs/gies.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/gies/{dag_model}_{is_scaled}_sim{sim}.benchmark.txt')
    shell:
        """
        Rscript workflow/scripts/gies.R --data {input} --out {output}
        """

rule run_notears_cv_std:
    input:
        data=join(std_config['output_dir'], 'data/{dag_model}/{is_scaled}/observational/sim_{sim}.npz'),
        cv_train=expand(join(std_config['output_dir'], 'data/{{dag_model}}/raw/observational/cv/sim{{sim}}/fold{fold}_train.npz'), fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'], 'data/{{dag_model}}/raw/observational/cv/sim{{sim}}/fold{fold}_val.npz'), fold=range(std_config['n_folds'])),
        lambda1=config['lambda_file'],
    output:
        join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/notears/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/notears/{is_scaled}/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{dag_model}/raw/observational/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/notears/{dag_model}_{is_scaled}_sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} --out {output} --lambdas {input.lambda1} --method notears --param_out {params.grid_out} --folds {params.n_folds}
        """

rule run_golem_ev_cv_std:
    input:
        data=join(std_config['output_dir'], 'data/{dag_model}/{is_scaled}/observational/sim_{sim}.npz'),
        lambda1=config['lambda_file'],
        lambda2=std_config['golem_lambda2_file'],
        cv_train=expand(join(std_config['output_dir'], 'data/{{dag_model}}/raw/observational/cv/sim{{sim}}/fold{fold}_train.npz'), fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'], 'data/{{dag_model}}/raw/observational/cv/sim{{sim}}/fold{fold}_val.npz'), fold=range(std_config['n_folds'])),
    output:
        join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/golem-ev/sim_{sim}.npy')
    conda:
        '../../workflow/envs/golem.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/golem-ev/{dag_model}_{is_scaled}_sim{sim}.benchmark.txt')
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/golem-ev/{is_scaled}/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{dag_model}/raw/observational/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} --out {output} --lambdas {input.lambda1} --lambda2s {input.lambda2} --method golem-ev --param_out {params.grid_out} --folds {params.n_folds}
        """

rule run_golem_nv_cv_std:
    input:
        data=join(std_config['output_dir'], 'data/{dag_model}/{is_scaled}/observational/sim_{sim}.npz'),
        lambda1=config['lambda_file'],
        lambda2=std_config['golem_lambda2_file'],
        cv_train=expand(join(std_config['output_dir'], 'data/{{dag_model}}/raw/observational/cv/sim{{sim}}/fold{fold}_train.npz'), fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'], 'data/{{dag_model}}/raw/observational/cv/sim{{sim}}/fold{fold}_val.npz'), fold=range(std_config['n_folds'])),
    output:
        join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/golem-nv/sim_{sim}.npy')
    conda:
        '../../workflow/envs/golem.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/golem-nv/{dag_model}_{is_scaled}_sim{sim}.benchmark.txt')
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/golem-nv/{is_scaled}/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{dag_model}/raw/observational/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} --out {output} --lambdas {input.lambda1} --lambda2s {input.lambda2} --method golem-nv --param_out {params.grid_out} --folds {params.n_folds}
        """

rule run_sortnregress_std:
    input:
        join(std_config['output_dir'], 'data/{dag_model}/{is_scaled}/observational/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/sortnregress/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/sortnregress/{dag_model}_{is_scaled}_sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/sortnregress.py --data {input}  --out {output} --use_lasso
        """

rule run_direct_lingam_std:
    input:
        join(std_config['output_dir'], 'data/{dag_model}/{is_scaled}/observational/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/direct-lingam/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/direct-lingam/{dag_model}_{is_scaled}_sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/run_naive.py --data {input} --method direct-lingam --out {output}
        """

rule convert_data_format_dcdi:
    input:
        expand(join(std_config['output_dir'], 'data/{{dag_model}}/{{is_scaled}}/interventional/sim_{sim}.npz'), sim=range(std_config['dcdi_n_sims']))
    output:
        expand(join(std_config['output_dir'], 'temp/dcdi/{{dag_model}}/{{is_scaled}}/interventional/data_interv{sim}.npy'), sim=range(std_config['dcdi_n_sims'])),
        expand(join(std_config['output_dir'], 'temp/dcdi/{{dag_model}}/{{is_scaled}}/interventional/regime{sim}.csv'), sim=range(std_config['dcdi_n_sims'])),
        expand(join(std_config['output_dir'], 'temp/dcdi/{{dag_model}}/{{is_scaled}}/interventional/intervention{sim}.csv'), sim=range(std_config['dcdi_n_sims'])),
        expand(join(std_config['output_dir'], 'temp/dcdi/{{dag_model}}/{{is_scaled}}/interventional/DAG{sim}.npy'), sim=range(std_config['dcdi_n_sims'])),
        expand(join(std_config['output_dir'], 'temp/dcdi/{{dag_model}}/{{is_scaled}}/interventional/CPDAG{sim}.npy'), sim=range(std_config['dcdi_n_sims']))
    params:
        in_dir=join(std_config['output_dir'],  'data/{dag_model}/{is_scaled}/interventional/'),
        out_dir=join(std_config['output_dir'], 'temp/dcdi/{dag_model}/{is_scaled}/interventional/'),
        dag_dir=join(std_config['output_dir'], 'dags/{dag_model}/'),
        n_sims=std_config['dcdi_n_sims'] 
    conda:
        '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/convert_data_to_dcdi_format.py --data {params.in_dir} --out {params.out_dir} --n_sims {params.n_sims} --dag {params.dag_dir}
        """

rule run_dcdi_std:
    input:
        join(std_config['output_dir'], 'temp/dcdi/{dag_model}/{is_scaled}/interventional/data_interv{sim}.npy'),
        join(std_config['output_dir'], 'temp/dcdi/{dag_model}/{is_scaled}/interventional/regime{sim}.csv'),
        join(std_config['output_dir'], 'temp/dcdi/{dag_model}/{is_scaled}/interventional/intervention{sim}.csv'),
        join(std_config['output_dir'], 'temp/dcdi/{dag_model}/{is_scaled}/interventional/DAG{sim}.npy'),
        join(std_config['output_dir'], 'temp/dcdi/{dag_model}/{is_scaled}/interventional/CPDAG{sim}.npy'),
    output:
        join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/dcdi-g/lambda{lambda1}/sim_{sim}.npy')
    params:
        in_dir=join(std_config['output_dir'], 'temp/dcdi/{dag_model}/{is_scaled}/interventional'),
        out_dir=join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/dcdi-g/out/sim{sim}/lambda{lambda1}'),
        out_dag=join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/dcdi-g/out/sim{sim}/lambda{lambda1}/train/DAG.npy'),
        p=std_config['p']
    conda:
        '../../workflow/envs/dcdi.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/dcdi-g/{dag_model}_{is_scaled}_sim{sim}_lambda{lambda1}.benchmark.txt')
    shell:
        """
        mkdir -p {params.out_dir} 
        module load R/4.1.0-BIO
        python workflow/scripts/dcdi/main.py --train --data-path {params.in_dir} --num-vars {params.p} --i-dataset {wildcards.sim} --exp-path {params.out_dir} --model DCDI-G --intervention --intervention-type perfect --intervention-knowledge known --reg-coeff {wildcards.lambda1}
        mv {params.out_dag} {output}
        """

rule run_igsp_std:
    input:
        join(std_config['output_dir'], 'data/{dag_model}/{is_scaled}/interventional/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/igsp/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    params:
        alpha=config['alpha'],
        alpha_inv=config['alpha_inv']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/igsp/{dag_model}_{is_scaled}_sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/igsp.py --data {input}  --out {output} --alpha {params.alpha} --alpha_inv {params.alpha_inv}
        """

rule run_ut_igsp_std:
    input:
        join(std_config['output_dir'], 'data/{dag_model}/{is_scaled}/interventional/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'out/{dag_model}/{is_scaled}/ut-igsp/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    params:
        alpha=config['alpha'],
        alpha_inv=config['alpha_inv']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/ut-igsp/{dag_model}_{is_scaled}_sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/ut_igsp.py --data {input}  --out {output} --alpha {params.alpha} --alpha_inv {params.alpha_inv}
        """
