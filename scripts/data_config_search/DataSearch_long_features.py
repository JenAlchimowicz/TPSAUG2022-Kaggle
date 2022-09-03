import mlflow

from search_pipeline import data_config_search


##############################
# Param setup
##############################

purpose = 'data_config_search2'
details = 'imputer_search'

imputer_methods = [
    {'method':'iterative', 'by_product_code':True}
]
attr2_types = [
    'float64'
]
attr3_types = [
    'float64'
]
group_features = [
    {'use':True}
]
poly_features = [
    {'use':False, 'degree':2}
]
spline_transformer = [
    {'use':True, 'n_knots':3, 'degree':3}
]
logic_features = [
    {'use':False}
]
loading_features = [
    {'use':False}
]
feature_scaling = [
    {'use':True, 'method':'standard_scaler'}
]
feature_selection_list = [
    ({'Constant_Features': {'frac_constant_values': 0.99}}, True)
]
models = [
    'logreg'
]

# Intermediary dict
seeds = [42, 16, 8]
param_dict = {
    'global': {'seeds': seeds},
    'feature_engineering': {
        'NA_flags': ['measurement_3', 'measurement_5'],
        'imputation': {
            'params': imputer_methods,
            'extra_params': {'seed':seeds[0], 'n_knn':3}},
        'attr_types': {'attribute_2':attr2_types, 'attribute_3':attr3_types},
        'group_features': group_features,
        'poly_features': poly_features,
        'spline_transformer': spline_transformer,
        'logic_features': logic_features,
        'loading_features': loading_features
    },
    'feature_scaling': feature_scaling,
    'feature_selection': feature_selection_list,
    'modeling':{
        'models': models
    }
}


##############################
# Main
##############################

if __name__ == '__main__':

    mlflow.set_tracking_uri("sqlite:///../../mlflow_data/mlflow.db")
    mlflow.set_experiment("TPSAUG2022-data-config")

    data_config_search(param_dict, purpose, details)