import numpy as np
import pandas as pd

import itertools
import os
import sys
import argparse

def convert_npz_to_dict(npz_file):
    out = {}
    for k, v in npz_file.items():
        out[k] = v
    return out

def concatenate_interventional_data(data):
    mats = [v for k, v in data.items()]
    return np.concatenate(mats).shape


def cv_dotears(lambda1, train_data, val_data, obs_only=False):
    from dotears import DOTEARS
    dotears_obj = DOTEARS(train_data, lambda1=lambda1, w_threshold=0, scaled=False, obs_only=obs_only)

    inferred_dag = dotears_obj.fit()
    # create a dotears object but do not fit
    dotears_obj_val = DOTEARS(val_data, lambda1=lambda1, w_threshold=0, scaled=False, obs_only=obs_only)
    val_loss, _ = dotears_obj_val.loss(inferred_dag)
    return val_loss

def cv_notears(lambda1, train_data, val_data):
    from notears import notears_linear, loss_notears
    lambda1 = params
    inferred_dag = notears_linear(train_data, lambda1=lambda1, w_threshold=0)
    val_loss, _ = loss_notears(inferred_dag, val_data)
    return val_loss

def cv_golem(lambda1, lambda2, train_data, val_data, equal_variances):
    sys.path.append('./workflow/scripts/golem/src')
    sys.path.append('./workflow/scripts/golem/src/models')
    sys.path.append('./workflow/scripts/golem/src/trainers')
    import golem
    from golem_trainer import GolemTrainer
    from golem_model import GolemModel
    import tensorflow as tf
    
    tf.compat.v1.disable_eager_execution() 
    W_train = golem.golem(np.copy(train_data), lambda_1=lambda1, lambda_2=lambda2, equal_variances=True)

    if not equal_variances:
        W_train = golem.golem(np.copy(train_data), lambda_1=lambda1, lambda_2=lambda2, equal_variances=False, B_init=W_train)

    n, p = val_data.shape
    
    model = GolemModel(n, p, lambda1, lambda2, equal_variances, W_train)
    trainer = GolemTrainer()
    model.sess.run(tf.compat.v1.global_variables_initializer())
    _, val_loss, _, _ = trainer.eval_iter(model, val_data)
    return val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_data', help='folder of data in cv format', type=str)
    parser.add_argument('--data', help='file of simulation data to run', type=str)
    parser.add_argument('--out', help='file to output inferred dag', type=str)
    parser.add_argument('--param_out', help='file for output cv grid', type=str)
    parser.add_argument('--lambdas', help='file with lambda options to cross-validate over', type=str)
    parser.add_argument('--lambda2s', type=str, help='golem only argument to file path with lambda2 options')
    parser.add_argument('--method', help='name of method to run', type=str)
    parser.add_argument('--folds', help='number of folds for cross validation', type=int)
    args = parser.parse_args()

    # cross validation
    lambdas = np.loadtxt(args.lambdas).astype(np.float64)
    method = args.method

    if method.startswith('golem'):
        lambda2s = np.loadtxt(args.lambda2s).astype(np.float64)
        param_set = itertools.product(lambdas, lambda2s)
        df = pd.DataFrame(columns=['loss', 'lambda', 'lambda2'])
    else:
        param_set = lambdas
        df = pd.DataFrame(columns=['loss', 'lambda'])

    for params in param_set:
        for i in range(args.folds):
            train_path = os.path.join(args.cv_data, 'fold{}_train.npz'.format(str(i)))
            val_path = os.path.join(args.cv_data, 'fold{}_val.npz'.format(str(i)))
            train_data = np.load(train_path)
            val_data = np.load(val_path)
            train_data = convert_npz_to_dict(train_data)
            val_data = convert_npz_to_dict(val_data)

            loss_per_split = np.asarray([])
            if method == 'dotears':
                val_loss = cv_dotears(params, train_data, val_data)

            if method == 'dotears_obsonly':
                val_loss = cv_dotears(params, train_data, val_data, obs_only=True)

            if method == 'notears':
                val_loss = cv_notears(params, train_data['obs'], val_data['obs'])

            if method.startswith('golem'):
                val_loss = cv_golem(params[0], params[1], train_data['obs'], val_data['obs'], equal_variances=(method == 'golem-ev'))

            if method == 'dcdi-g':
                pass

            if method == 'notears_interventional':
                train_data_all = concatenate_interventional_data(train_data)
                val_data_all = concatenate_interventional_data(val_data)
                val_loss = cv_notears(params, train_data_all, val_data_all)
    
            loss_per_split = np.append(loss_per_split, val_loss)

        param_loss = np.mean(loss_per_split)
        if method.startswith('golem'):
            df.loc[len(df.index)] = [param_loss, params[0], params[1]]
        else:
            df.loc[len(df.index), :] = [param_loss, params]
                  
            
    best_performing_parameters = df[df['loss'] == df['loss'].min()]
    data = np.load(args.data)
    if method == 'dotears':
        from dotears import DOTEARS
        lambda1 = best_performing_parameters['lambda'].values[0]
        data = convert_npz_to_dict(data)
        dotears_obj = DOTEARS(data,
                              lambda1=lambda1,
                              scaled=False,
                              w_threshold=0)
        inferred_dag = dotears_obj.fit()
    if method == 'dotears_obsonly':
        from dotears import DOTEARS
        lambda1 = best_performing_parameters['lambda'].values[0]
        data = convert_npz_to_dict(data)
        dotears_obj = DOTEARS(data,
                              lambda1=lambda1,
                              scaled=False,
                              w_threshold=0,
                              obs_only=True)
        inferred_dag = dotears_obj.fit()
    if method == 'notears':
        from notears import notears_linear
        lambda1 = best_performing_parameters['lambda'].values[0]
        inferred_dag = notears_linear(data['obs'],
                       lambda1=lambda1,
                       w_threshold=0)
    if method == 'notears_interventional':
        from notears import notears_linear
        lambda1 = best_performing_parameters['lambda'].values[0]
        data_all = concatenate_interventional_data(data)
        inferred_dag = notears_linear(data['obs'],
                       lambda1=lambda1,
                       w_threshold=0)

    if method.startswith('golem'):
        sys.path.append('./workflow/scripts/golem/src')
        sys.path.append('./workflow/scripts/golem/src/models')
        sys.path.append('./workflow/scripts/golem/src/trainers')
        import golem
        from golem_trainer import GolemTrainer
        from golem_model import GolemModel
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        lambda1 = best_performing_parameters['lambda'].values[0]
        lambda2 = best_performing_parameters['lambda2'].values[0]

        inferred_dag = golem.golem(np.copy(data['obs']), lambda_1=lambda1, lambda_2=lambda2, equal_variances=True)
        if (method == 'golem-nv'):
            inferred_dag = golem.golem(np.copy(data['obs']), lambda_1=lambda1, lambda_2=lambda2, equal_variances=False, B_init=inferred_dag)

    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out))
    np.save(args.out, inferred_dag)

    if not os.path.exists(os.path.dirname(args.param_out)):
        os.makedirs(os.path.dirname(args.param_out))
    df.to_csv(args.param_out)    
