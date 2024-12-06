rule dotears:
    input:
        data=join(std_config['output_dir'], 'data/{prefix}/interventional/sim_{sim}.npz'),
        cv_train=expand(join(std_config['output_dir'], 
                        'data/{{prefix}}/interventional/cv/sim{{sim}}/fold{fold}_train.npz'), 
                        fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'],
                      'data/{{prefix}}/interventional/cv/sim{{sim}}/fold{fold}_val.npz'),
                      fold=range(std_config['n_folds'])),
        lambda1=config['lambda_file'],
    output:
        join(std_config['output_dir'], 'out/{prefix}/dotears/sim_{sim}.npy')
    conda:
        '../../workflow/envs/dotears.yml'
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/dotears/{prefix}/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{prefix}/interventional/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/dotears/{prefix}/sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} \
            --data {input.data} --out {output} --lambdas {input.lambda1} --method dotears \
            --param_out {params.grid_out} --folds {params.n_folds}
        """

rule dotears_obsonly:
    input:
        data=join(std_config['output_dir'], 'data/{prefix}/interventional/sim_{sim}.npz'),
        cv_train=expand(join(std_config['output_dir'],
                       'data/{{prefix}}/interventional/cv/sim{{sim}}/fold{fold}_train.npz'), 
                       fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'], 
                      'data/{{prefix}}/interventional/cv/sim{{sim}}/fold{fold}_val.npz'), 
                      fold=range(std_config['n_folds'])),
        lambda1=config['lambda_file'],
    output:
        join(std_config['output_dir'], 'out/{prefix}/dotears_obsonly/sim_{sim}.npy')
    conda:
        '../../workflow/envs/dotears.yml'
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/{prefix}/dotears_obsonly/raw/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{prefix}/interventional/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/dotears_obsonly/{prefix}/sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} \
        --out {output} --lambdas {input.lambda1} --method dotears_obsonly --param_out {params.grid_out} \
        --folds {params.n_folds}
        """

rule gies:
    input:
        join(std_config['output_dir'], 'data/{prefix}/interventional/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'out/{prefix}/gies/sim_{sim}.npy')
    conda:
        '../../workflow/envs/gies.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/gies/{prefix}/sim{sim}.benchmark.txt')
    shell:
        """
        Rscript workflow/scripts/gies.R --data {input} --out {output}
        """

rule notears:
    input:
        data=join(std_config['output_dir'], 'data/{prefix}/observational/sim_{sim}.npz'),
        cv_train=expand(join(std_config['output_dir'], 
                        'data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_train.npz'), 
                        fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'], 
                      'data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_val.npz'), 
                      fold=range(std_config['n_folds'])),
        lambda1=config['lambda_file'],
    output:
        join(std_config['output_dir'], 'out/{prefix}/notears/sim_{sim}.npy')
    conda:
        '../../workflow/envs/dotears.yml'
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/notears/{prefix}/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{prefix}/observational/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/notears/{prefix}/sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} \
        --out {output} --lambdas {input.lambda1} --method notears --param_out {params.grid_out} \
        --folds {params.n_folds}
        """

rule golem_ev:
    input:
        data=join(std_config['output_dir'], 'data/{prefix}/observational/sim_{sim}.npz'),
        lambda1=config['lambda_file'],
        lambda2=std_config['golem_lambda2_file'],
        cv_train=expand(join(std_config['output_dir'], 
                        'data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_train.npz'), 
                        fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'], 'data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_val.npz'), fold=range(std_config['n_folds'])),
    output:
        join(std_config['output_dir'], 'out/{prefix}/golem-ev/sim_{sim}.npy')
    conda:
        '../../workflow/envs/golem.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/golem-ev/{prefix}/sim{sim}.benchmark.txt')
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/golem-ev/{prefix}/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{prefix}/observational/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} --out {output} \
        --lambdas {input.lambda1} --lambda2s {input.lambda2} --method golem-ev --param_out {params.grid_out} \
        --folds {params.n_folds}
        """

rule golem_nv:
    input:
        data=join(std_config['output_dir'], 'data/{prefix}/observational/sim_{sim}.npz'),
        lambda1=config['lambda_file'],
        lambda2=std_config['golem_lambda2_file'],
        cv_train=expand(join(std_config['output_dir'], 
                        'data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_train.npz'), 
                        fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'], 
                      'data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_val.npz'), 
                      fold=range(std_config['n_folds'])),
    output:
        join(std_config['output_dir'], 'out/{prefix}/golem-nv/sim_{sim}.npy')
    conda:
        '../../workflow/envs/golem.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/golem-nv/{prefix}/sim{sim}.benchmark.txt')
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/golem-nv/{prefix}/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{prefix}/observational/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} --data {input.data} \
        --out {output} --lambdas {input.lambda1} --lambda2s {input.lambda2} --method golem-nv --param_out {params.grid_out} \
        --folds {params.n_folds}
        """

rule sortnregress:
    input:
        join(std_config['output_dir'], 'data/{prefix}/observational/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'out/{prefix}/sortnregress/sim_{sim}.npy')
    conda:
        '../../workflow/envs/dotears.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/sortnregress/{prefix}/sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/sortnregress.py --data {input}  --out {output} --use_lasso
        """

rule direct_lingam:
    input:
        join(std_config['output_dir'], 'data/{prefix}/observational/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'out/{prefix}/direct-lingam/sim_{sim}.npy')
    conda:
        '../../workflow/envs/lingam.yml'
    benchmark:
        join(std_config['output_dir'], 'benchmarks/direct-lingam/{prefix}/sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/run_naive.py --data {input} --method direct-lingam --out {output}
        """


rule igsp:
    input:
        join(std_config['output_dir'], 'data/{prefix}/interventional/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'out/{prefix}/igsp/sim_{sim}.npy')
    conda:
        '../../workflow/envs/dotears.yml'
    params:
        alpha=config['alpha'],
        alpha_inv=config['alpha_inv']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/igsp/{prefix}/sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/igsp.py --data {input}  --out {output} --alpha {params.alpha} --alpha_inv {params.alpha_inv}
        """

rule ut_igsp:
    input:
        join(std_config['output_dir'], 'data/{prefix}/interventional/sim_{sim}.npz')
    output:
        join(std_config['output_dir'], 'out/{prefix}/ut-igsp/sim_{sim}.npy')
    conda:
        '../../workflow/envs/dotears.yml'
    params:
        alpha=config['alpha'],
        alpha_inv=config['alpha_inv']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/ut-igsp/{prefix}/sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/ut_igsp.py --data {input}  --out {output} --alpha {params.alpha} --alpha_inv {params.alpha_inv}
        """

rule colide_nv:
    input:
        data=join(std_config['output_dir'], 'data/{prefix}/observational/sim_{sim}.npz'),
        cv_train=expand(join(std_config['output_dir'], 
                        'data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_train.npz'), 
                        fold=range(std_config['n_folds'])),
        cv_val=expand(join(std_config['output_dir'],
                      'data/{{prefix}}/observational/cv/sim{{sim}}/fold{fold}_val.npz'),
                      fold=range(std_config['n_folds'])),
        lambda1=config['lambda_file'],
    output:
        join(std_config['output_dir'], 'out/{prefix}/colide-nv/sim_{sim}.npy')
    conda:
        '../../workflow/envs/colide-nv.yml'
    params:
        grid_out=join(std_config['output_dir'], 'data/param_grid/colide-nv/{prefix}/sim_{sim}.csv'),
        cv_folder=join(std_config['output_dir'], 'data/{prefix}/observational/cv/sim{sim}'),
        n_folds=std_config['n_folds']
    benchmark:
        join(std_config['output_dir'], 'benchmarks/colide-nv/{prefix}/sim{sim}.benchmark.txt')
    shell:
        """
        python workflow/scripts/cross_validation.py --cv_data {params.cv_folder} \
            --data {input.data} --out {output} --lambdas {input.lambda1} --method colide-nv \
            --param_out {params.grid_out} --folds {params.n_folds}
        """
