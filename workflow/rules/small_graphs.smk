# load small_configfile

rule gen_interventional_small_graph_data:
    input:
        w=join(small_config['output_dir'], 'dags', '{dag_type}.txt'),
        beta_file=small_config['beta_file'],
        sigma_1_file=small_config['sigma_1_file']
    output:
        int_out=expand(join(small_config['output_dir'], 'data/interventional/{{dag_type}}/sigma1_{sigma1}/beta_{beta}', 'sim_{sim}.npz'), 
                    sigma1=SMALL_SIGMA_1S, 
                    beta=SMALL_BETAS,
                    sim=range(small_config['n_sims'])),
        obs_out=expand(join(small_config['output_dir'], 'data/observational/{{dag_type}}/sigma1_{sigma1}/beta_{beta}', 'sim_{sim}.npz'),
                    sigma1=SMALL_SIGMA_1S,
                    beta=SMALL_BETAS,
                    sim=range(small_config['n_sims']))
    params:
        n=small_config['n'],
        a=small_config['a'],
        n_sims=small_config['n_sims'],
        out_parent=join(small_config['output_dir'], 'data') 
    conda:
        '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/gen_small_graph_data.py --w_template {input.w} --sigma_1 {input.sigma_1_file} --beta {input.beta_file} --n {params.n} --n_sims {params.n_sims} --a {params.a} --out {params.out_parent}
        python workflow/scripts/gen_small_graph_data.py --w_template {input.w} --sigma_1 {input.sigma_1_file} --beta {input.beta_file} --n {params.n} --n_sims {params.n_sims} --a {params.a} --out {params.out_parent} --observational_data
        """

rule run_dotears_naive:
    input:
        join(small_config['output_dir'], 'data/interventional/{parent_dir}/sim_{sim}.npz')
    output:
        join(small_config['output_dir'], 'out/dotears/{parent_dir}/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/run_naive.py --data {input} --method DOTEARS --out {output}
        """

rule run_gies_naive:
    input:
        join(small_config['output_dir'], 'data/interventional/{parent_dir}/sim_{sim}.npz')
    output:
        join(small_config['output_dir'], 'out/gies/{parent_dir}/sim_{sim}.npy')
    conda:
        '../../workflow/envs/gies.yml'
    shell:
        """
        Rscript workflow/scripts/gies.R --data {input} --out {output}
        """

rule run_notears_naive:
    input:
        join(small_config['output_dir'], 'data/observational/{parent_dir}/sim_{sim}.npz')
    output:
        join(small_config['output_dir'], 'out/notears/{parent_dir}/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/run_naive.py --data {input} --method NOTEARS --out {output}
        """

rule run_sortnregress_naive:
    input:
        join(small_config['output_dir'], 'data/observational/{parent_dir}/sim_{sim}.npz')
    output:
        join(small_config['output_dir'], 'out/sortnregress/{parent_dir}/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/run_naive.py --data {input} --method sortnregress --out {output}
        """

rule run_golem_ev_naive:
    input:
        join(small_config['output_dir'], 'data/observational/{parent_dir}/sim_{sim}.npz')
    output:
        join(small_config['output_dir'], 'out/golem-ev/{parent_dir}/sim_{sim}.npy')
    conda:
        '../../workflow/envs/golem.yml'
    shell:
        """
        python workflow/scripts/run_naive.py --data {input} --method GOLEM-EV --out {output}
        """

rule run_golem_nv_naive:
    input:
        join(small_config['output_dir'], 'data/observational/{parent_dir}/sim_{sim}.npz')
    output:
        join(small_config['output_dir'], 'out/golem-nv/{parent_dir}/sim_{sim}.npy')
    conda:
        '../../workflow/envs/golem.yml'
    shell:
        """
        python workflow/scripts/run_naive.py --data {input} --method GOLEM-NV --out {output}
        """

rule run_direct_lingam_naive:
    input:
        join(small_config['output_dir'], 'data/observational/{parent_dir}/sim_{sim}.npz')
    output:
        join(small_config['output_dir'], 'out/direct-lingam/{parent_dir}/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/run_naive.py --data {input} --method direct-lingam --out {output}
        """

rule run_igsp_naive:
    input:
        join(small_config['output_dir'], 'data/interventional/{parent_dir}/sim_{sim}.npz')
    output:
        join(small_config['output_dir'], 'out/igsp/{parent_dir}/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    params:
        alpha=config['alpha'],
        alpha_inv=config['alpha_inv']
    shell:
        """
        python workflow/scripts/igsp.py --data {input} --out {output} --alpha {params.alpha} --alpha_inv {params.alpha_inv}
        """

rule run_ut_igsp_naive:
    input:
        join(small_config['output_dir'], 'data/interventional/{parent_dir}/sim_{sim}.npz')
    output:
        join(small_config['output_dir'], 'out/ut-igsp/{parent_dir}/sim_{sim}.npy')
    conda:
        '../../workflow/envs/std.yml'
    params:
        alpha=config['alpha'],
        alpha_inv=config['alpha_inv']
    shell:
        """
        python workflow/scripts/ut_igsp.py --data {input} --out {output} --alpha {params.alpha} --alpha_inv {params.alpha_inv}
        """

rule convert_data_format_dcdi_small:
    input:
        expand(join(small_config['output_dir'], 'data/interventional/{{dag}}/sigma1_{{sigma}}/beta_{{beta}}/sim_{sim}.npz'), sim=range(small_config['n_sims']))
    output:
        expand(join(small_config['output_dir'], 'temp/dcdi/interventional/{{dag}}/sigma1_{{sigma}}/beta_{{beta}}/data_interv{sim}.npy'), sim=range(small_config['n_sims'])),
        expand(join(small_config['output_dir'], 'temp/dcdi/interventional/{{dag}}/sigma1_{{sigma}}/beta_{{beta}}/regime{sim}.csv'), sim=range(small_config['n_sims'])),
        expand(join(small_config['output_dir'], 'temp/dcdi/interventional/{{dag}}/sigma1_{{sigma}}/beta_{{beta}}/intervention{sim}.csv'), sim=range(small_config['n_sims'])),
        expand(join(small_config['output_dir'], 'temp/dcdi/interventional/{{dag}}/sigma1_{{sigma}}/beta_{{beta}}/DAG{sim}.npy'), sim=range(small_config['n_sims'])),
        expand(join(small_config['output_dir'], 'temp/dcdi/interventional/{{dag}}/sigma1_{{sigma}}/beta_{{beta}}/CPDAG{sim}.npy'), sim=range(small_config['n_sims']))
    params:
        in_dir=join(small_config['output_dir'],  'data/interventional/{dag}/sigma1_{sigma}/beta_{beta}/'),
        out_dir=join(small_config['output_dir'], 'temp/dcdi/interventional/{dag}/sigma1_{sigma}/beta_{beta}/'),
        dag_path=join(small_config['output_dir'], 'dags/{dag}.txt'),
        n_sims=small_config['n_sims']
    conda:
        '../../workflow/envs/std.yml'
    shell:
        """
        python workflow/scripts/convert_data_to_dcdi_format.py --data {params.in_dir} --out {params.out_dir} --n_sims {params.n_sims} --small_dag_path {params.dag_path} --small
        """

rule run_dcdi_small:
    input:
        join(small_config['output_dir'], 'temp/dcdi/interventional/{dag}/sigma1_{sigma}/beta_{beta}/data_interv{sim}.npy'),
        join(small_config['output_dir'], 'temp/dcdi/interventional/{dag}/sigma1_{sigma}/beta_{beta}/regime{sim}.csv'),
        join(small_config['output_dir'], 'temp/dcdi/interventional/{dag}/sigma1_{sigma}/beta_{beta}/intervention{sim}.csv'),
        join(small_config['output_dir'], 'temp/dcdi/interventional/{dag}/sigma1_{sigma}/beta_{beta}/DAG{sim}.npy'),
        join(small_config['output_dir'], 'temp/dcdi/interventional/{dag}/sigma1_{sigma}/beta_{beta}/CPDAG{sim}.npy'),
    output:
        join(small_config['output_dir'], 'out/dcdi-g/{dag}/sigma1_{sigma}/beta_{beta}/sim_{sim}.npy')
    params:
        in_dir=join(small_config['output_dir'], 'temp/dcdi/interventional/{dag}/sigma1_{sigma}/beta_{beta}/'),
        out_dir=join(small_config['output_dir'], 'out/dcdi-g/out/{dag}/sigma1_{sigma}/beta_{beta}/sim{sim}/'),
        out_dag=join(small_config['output_dir'], 'out/dcdi-g/out/{dag}/sigma1_{sigma}/beta_{beta}/sim{sim}/train/DAG.npy'),
        p=lambda wc: wc.dag.split('_')[-1]
    conda:
        '../../workflow/envs/dcdi.yml'
    shell:
        """
        mkdir -p {params.out_dir}
        module load R/4.1.0-BIO
        python workflow/scripts/dcdi/main.py --train --data-path {params.in_dir} --num-vars {params.p} --i-dataset {wildcards.sim} --exp-path {params.out_dir} --model DCDI-G --intervention --intervention-type perfect --intervention-knowledge known --reg-coeff 0
        mv {params.out_dag} {output}
        """
