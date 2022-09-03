import pandas as pd
import numpy as np
import itertools
import mlflow

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import catboost as cb

from data_transform_utils import fill_missing_values, assign_attr_types, add_NA_flags, one_hot_encode
from data_transform_utils import add_logic_features, add_poly, add_spline, add_loading_features, add_group_features
from data_transform_utils import scale, feature_selection


def data_config_search(param_dict, purpose, details):

    #######################################
    # Create iterator & model dict
    #######################################

    # This solution is more readable than several for loops
    iterator = itertools.product(
        param_dict['feature_engineering']['imputation']['params'],
        param_dict['feature_engineering']['attr_types']['attribute_2'],
        param_dict['feature_engineering']['attr_types']['attribute_3'],
        param_dict['feature_engineering']['group_features'],
        param_dict['feature_engineering']['poly_features'],
        param_dict['feature_engineering']['spline_transformer'],
        param_dict['feature_engineering']['logic_features'],
        param_dict['feature_engineering']['loading_features'],
        param_dict['feature_scaling'],
        param_dict['feature_selection'],
        param_dict['modeling']['models']
    )


    #######################################
    # Run pipeline
    #######################################
    for elements in iterator:

        imputer_params = elements[0]
        attr2_type = elements[1]
        attr3_type = elements[2]
        group_dict = elements[3]
        poly_dict = elements[4]
        spline_dict = elements[5]
        logic_dict = elements[6]
        loading_dict = elements[7]
        scaling_dict = elements[8]
        fselection_steps, stepwise = elements[9]
        model = elements[10]

        with mlflow.start_run():
            mlflow.set_tag('purpose', purpose) # e.g. data param search
            mlflow.set_tag('details', details) # e.g. imputer search
            mlflow.set_tag('data_version', 'v1')
            mlflow.set_tag('code_version', 'v1')

            mlflow.log_param('imputer_method', imputer_params['method'])
            mlflow.log_param('imputer_by_prod_code', imputer_params['by_product_code'])
            mlflow.log_param('attr2_type', attr2_type)
            mlflow.log_param('attr3_type', attr3_type)
            mlflow.log_param('group_features_used', group_dict['use'])
            mlflow.log_param('poly_features_used', poly_dict['use'])
            mlflow.log_param('poly_degree', poly_dict['degree'])
            mlflow.log_param('spline_params_used', spline_dict['use'])
            mlflow.log_param('spline_n_knots', spline_dict['n_knots'])
            mlflow.log_param('spline_degree', spline_dict['degree'])
            mlflow.log_param('logic_features_used', logic_dict['use'])
            mlflow.log_param('loading_features_used', loading_dict['use'])
            mlflow.log_param('scaling_used', scaling_dict['use'])
            mlflow.log_param('scaling_method', scaling_dict['method'])
            mlflow.log_params({'fs_'+k:True for k,v in fselection_steps.items()})
            mlflow.log_param('fs_stepwise_used', stepwise)
            mlflow.log_param('model', model)
            mlflow.log_param('seed', str(param_dict['global']['seeds']))

            ################
            # Prepare data
            ################
            train = pd.read_csv('../../data/train.csv', index_col='id')
            data = train.drop(columns=['failure'])

            data = (data
                .pipe(add_NA_flags, cols=param_dict['feature_engineering']['NA_flags'])
                .pipe(fill_missing_values, params=imputer_params, extra_params=param_dict['feature_engineering']['imputation']['extra_params'])
                .pipe(assign_attr_types, attr_types={'attribute_2':attr2_type, 'attribute_3':attr3_type})
                .pipe(add_group_features, group_dict=group_dict)
                .pipe(one_hot_encode)
                .pipe(add_poly, poly_dict=poly_dict)
                .pipe(add_spline, spline_dict=spline_dict)
                .pipe(add_logic_features, logic_dict=logic_dict)
                .pipe(add_loading_features, loading_dict=loading_dict)
                .pipe(scale, scale_dict=scaling_dict)
                .pipe(feature_selection, y=train['failure'], steps=fselection_steps, stepwise=stepwise, seed=param_dict['global']['seeds'][0])
                )
                

            ################
            # Get auroc
            ################
            auroc_dict = {'auroc_fold_A': 0, 'auroc_fold_B': 0, 'auroc_fold_C': 0, 'auroc_fold_D': 0, 'auroc_fold_E': 0}
            accuracy_list = []

            for seed in param_dict['global']['seeds']:

                # Use 5-fold split
                kfold = GroupKFold(n_splits=5)
                X = data
                y = train['failure']
                for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y, train['product_code'])):
                        
                    X_train = X.loc[train_idx]
                    X_val = X.loc[val_idx]
                    y_train = y.loc[train_idx]
                    y_val = y.loc[val_idx]

                    val_prod_code = X_val['product_code'].unique()[0]
                    X_train = X_train.drop(columns=['product_code'])
                    X_val = X_val.drop(columns=['product_code'])

                    # model_ = model_dict[model]
                    model_ = LogisticRegression(random_state=seed, max_iter=215, C=1.67)
                    model_.fit(X_train, y_train)
                    y_pred = model_.predict_proba(X_val)[:,1]

                    auroc_dict[f'auroc_fold_{val_prod_code}'] += roc_auc_score(y_val, y_pred)
                    accuracy_list.append(np.mean((y_pred >= 0.5).astype('int') == y_val))
            
            auroc_dict = {k:v/len(param_dict['global']['seeds']) for k,v in auroc_dict.items()}
            mlflow.log_param('n_features_used', data.shape[1])
            mlflow.log_metric('accuracy', np.mean(accuracy_list)*100)
            mlflow.log_metric('auroc', np.mean(list(auroc_dict.values()))*100)
            mlflow.log_metrics({k:v*100 for k,v in auroc_dict.items()})
