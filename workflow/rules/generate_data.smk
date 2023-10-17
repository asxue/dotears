rule generate_dags:
    output:
        er=expand(join(std_config['output_dir'], 'dags/{{experiment}}/erdos_renyi/sim_{sim}.txt'),
                 sim=range(std_config['n_sims'])),
        sf=expand(join(std_config['output_dir'], 'dags/{{experiment}}/scale_free/sim_{sim}.txt'),
                 sim=range(std_config['n_sims']))
    params:
        p=std_config['p'],
        d=std_config['d'],
        k=std_config['k'],
        lower_edge_range=std_config['lower_edge_range'],
        upper_edge_range=std_config['upper_edge_range'],
        n_sims=std_config['n_sims'],
        out=join(std_config['output_dir'], 'dags/{experiment}')
    conda:
        '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/erdos_renyi.py --p {params.p} --d {params.d} --k {params.k} --out {params.out} \
        --lower_edge_range {params.lower_edge_range} --upper_edge_range {params.upper_edge_range} \
        --n_sims {params.n_sims}
        """

rule generate_standard_data:
    input:
        join(std_config['output_dir'], 'dags/standard/{dag_model}/sim_{sim}.txt')
    output:
        out_obs=join(std_config['output_dir'], 'data/standard/{dag_model}/observational/sim_{sim}.npz'),
        out_int=join(std_config['output_dir'], 'data/standard/{dag_model}/interventional/sim_{sim}.npz'),
        out_obs_cv=join(std_config['output_dir'], 'data/standard/{dag_model}/observational/cv/sim_{sim}.npz'),
        out_int_cv=join(std_config['output_dir'], 'data/standard/{dag_model}/interventional/cv/sim_{sim}.npz')
    params:
        n=std_config['n'],
        std_lower_range=std_config['lower_std_range'],
        std_upper_range=std_config['upper_std_range'],
        a=std_config['a'],
        cv_seed='{sim}' + str(std_config['n_sims']) # not beautiful but deterministic
    conda:
       '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/sem.py --out {output.out_obs} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --a {params.a} --seed {wildcards.sim} --type observational 
        python workflow/scripts/sem.py --out {output.out_obs_cv} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --a {params.a} --seed {params.cv_seed} --type observational
        python workflow/scripts/sem.py --out {output.out_int} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --a {params.a} --seed {wildcards.sim} --type interventional 
        python workflow/scripts/sem.py --out {output.out_int_cv} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --a {params.a} --seed {params.cv_seed} --type interventional 
        """

rule generate_fixed_intervention_data:
    input:
        join(std_config['output_dir'], 'dags/fixed_intervention/{dag_model}/sim_{sim}.txt')
    output:
        out_obs=join(std_config['output_dir'], 'data/fixed_intervention/{dag_model}/observational/sim_{sim}.npz'),
        out_int=join(std_config['output_dir'], 'data/fixed_intervention/{dag_model}/interventional/sim_{sim}.npz'),
        out_obs_cv=join(std_config['output_dir'], 'data/fixed_intervention/{dag_model}/observational/cv/sim_{sim}.npz'),
        out_int_cv=join(std_config['output_dir'], 'data/fixed_intervention/{dag_model}/interventional/cv/sim_{sim}.npz')
    params:
        n=std_config['n'],
        std_lower_range=std_config['lower_std_range'],
        std_upper_range=std_config['upper_std_range'],
        cv_seed='{sim}' + str(std_config['n_sims']) + '0' # not beautiful but deterministic
    conda:
       '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/fixed_intervention_sem.py --out {output.out_obs} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --seed {wildcards.sim} --type observational 
        python workflow/scripts/fixed_intervention_sem.py --out {output.out_obs_cv} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --seed {params.cv_seed} --type observational
        python workflow/scripts/fixed_intervention_sem.py --out {output.out_int} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --seed {wildcards.sim} --type interventional 
        python workflow/scripts/fixed_intervention_sem.py --out {output.out_int_cv} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --seed {params.cv_seed} --type interventional 
        """

rule generate_parental_influence_data:
    input:
        join(std_config['output_dir'], 'dags/pi_{pi}/{dag_model}/sim_{sim}.txt')
    output:
        out_obs=join(std_config['output_dir'], 'data/pi_{pi}/{dag_model}/observational/sim_{sim}.npz'),
        out_int=join(std_config['output_dir'], 'data/pi_{pi}/{dag_model}/interventional/sim_{sim}.npz'),
        out_obs_cv=join(std_config['output_dir'], 'data/pi_{pi}/{dag_model}/observational/cv/sim_{sim}.npz'),
        out_int_cv=join(std_config['output_dir'], 'data/pi_{pi}/{dag_model}/interventional/cv/sim_{sim}.npz')
    params:
        n=std_config['n'],
        std_lower_range=std_config['lower_std_range'],
        std_upper_range=std_config['upper_std_range'],
        a=std_config['a'],
        cv_seed=lambda wildcards: wildcards.sim + str(std_config['n_sims']) + str(int(100 * float(wildcards.pi))) # not beautiful but deterministic
 # not beautiful but deterministic
    conda:
       '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/sem.py --out {output.out_obs} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --a {params.a} --seed {wildcards.sim} --type observational --parental_influence {wildcards.pi}
        python workflow/scripts/sem.py --out {output.out_obs_cv} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --a {params.a} --seed {params.cv_seed} --type observational --parental_influence {wildcards.pi}
        python workflow/scripts/sem.py --out {output.out_int} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --a {params.a} --seed {wildcards.sim} --type interventional --parental_influence {wildcards.pi}
        python workflow/scripts/sem.py --out {output.out_int_cv} --b_template {input} --random_var \
        --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} --n {params.n} \
        --a {params.a} --seed {params.cv_seed} --type interventional --parental_influence {wildcards.pi}
        """

rule generate_low_sample_size_data:
    input:
        join(std_config['output_dir'], 'dags/low_sample_size/{dag_model}/sim_{sim}.txt')
    output:
        out_obs=join(std_config['output_dir'], 'data/low_sample_size/{dag_model}/observational/sim_{sim}.npz'),
        out_int=join(std_config['output_dir'], 'data/low_sample_size/{dag_model}/interventional/sim_{sim}.npz'),
        out_obs_cv=join(std_config['output_dir'], 'data/low_sample_size/{dag_model}/observational/cv/sim_{sim}.npz'),
        out_int_cv=join(std_config['output_dir'], 'data/low_sample_size/{dag_model}/interventional/cv/sim_{sim}.npz')
    params:
        n=10,
        std_lower_range=std_config['lower_std_range'],
        std_upper_range=std_config['upper_std_range'],
        a=std_config['a'],
        cv_seed='{sim}' + str(std_config['n_sims']) + '1' # not beautiful but deterministic
    conda:
       '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/sem.py --out {output.out_obs} --b_template {input} --random_var \
            --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} \
            --n {params.n} --a {params.a} --seed {wildcards.sim} --type observational
        python workflow/scripts/sem.py --out {output.out_obs_cv} --b_template {input} --random_var \
            --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} \
            --n {params.n} --a {params.a} --seed {params.cv_seed} --type observational
        python workflow/scripts/sem.py --out {output.out_int} --b_template {input} --random_var \
            --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} \
            --n {params.n} --a {params.a} --seed {wildcards.sim} --type interventional
        python workflow/scripts/sem.py --out {output.out_int_cv} --b_template {input} --random_var \
            --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} \
            --n {params.n} --a {params.a} --seed {params.cv_seed} --type interventional
        """

rule generate_alpha_perturb_data:
    input:
        join(std_config['output_dir'], 'dags/a_perturbed/{dag_model}/sim_{sim}.txt')
    output:
        out_obs=join(std_config['output_dir'], 'data/a_perturbed/{dag_model}/observational/sim_{sim}.npz'),
        out_int=join(std_config['output_dir'], 'data/a_perturbed/{dag_model}/interventional/sim_{sim}.npz'),
        out_obs_cv=join(std_config['output_dir'], 'data/a_perturbed/{dag_model}/observational/cv/sim_{sim}.npz'),
        out_int_cv=join(std_config['output_dir'], 'data/a_perturbed/{dag_model}/interventional/cv/sim_{sim}.npz')
    params:
        n=std_config['n'],
        std_lower_range=std_config['lower_std_range'],
        std_upper_range=std_config['upper_std_range'],
        a=std_config['a'],
        cv_seed='{sim}' + str(std_config['n_sims']) + '2'# not beautiful but deterministic
    conda:
       '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/sem.py --out {output.out_obs} --b_template {input} --random_var \
            --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} \
            --n {params.n} --a {params.a} --seed {wildcards.sim} --type observational --a_perturbation
        python workflow/scripts/sem.py --out {output.out_obs_cv} --b_template {input} --random_var \
            --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} \
            --n {params.n} --a {params.a} --seed {params.cv_seed} --type observational --a_perturbation
        python workflow/scripts/sem.py --out {output.out_int} --b_template {input} --random_var \
            --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} \
            --n {params.n} --a {params.a} --seed {wildcards.sim} --type interventional --a_perturbation
        python workflow/scripts/sem.py --out {output.out_int_cv} --b_template {input} --random_var \
            --std_lower_range {params.std_lower_range} --std_upper_range {params.std_upper_range} \
            --n {params.n} --a {params.a} --seed {params.cv_seed} --type interventional --a_perturbation
        """

rule write_cv_folds:
    input:
        join(std_config['output_dir'], 'data/{prefix}/cv/sim_{sim}.npz')    
    output:
        expand(join(std_config['output_dir'],
                    'data/{{prefix}}/cv/sim{{sim}}/fold{fold}_train.npz'),
                    fold=range(std_config['n_folds'])),
        expand(join(std_config['output_dir'], 
                    'data/{{prefix}}/cv/sim{{sim}}/fold{fold}_val.npz'), 
                    fold=range(std_config['n_folds'])),
    params:
        out_dir=join(std_config['output_dir'], 'data/{prefix}/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    conda:
        '../../workflow/envs/dotears.yml'
    shell:
        """
        python workflow/scripts/convert_data_to_cv_format.py --data {input} --out_folder {params.out_dir} \
        --folds {params.n_folds} --random_state {wildcards.sim} --sim {wildcards.sim}
        """