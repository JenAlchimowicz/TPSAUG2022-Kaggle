import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from feature_selction_utils import FeatureSelector


############################
# Fill missing values
############################

def fill_missing_values(df:pd.DataFrame, params:dict, extra_params:dict):

    method = params['method']
    by_product_code = params['by_product_code']

    # Check validity of call
    if len(method)==0: raise ValueError('No imputation method specified - there has to be one')
    if method not in ['mean', 'knn', 'iterative']: raise ValueError(f'Method {method} not defined')
    
    method_dict = {
        'mean': SimpleImputer(strategy='mean'),
        'knn': KNNImputer(n_neighbors=extra_params['n_knn']),
        'iterative': IterativeImputer(random_state=extra_params['seed'])
    }

    float_cols = [f for f in df.columns if df[f].dtype == float]

    # Loading uncorrelated with any other feature -> fill with mean
    df['loading'].fillna(df['loading'].mean(), inplace=True)

    # Rest of columns
    if not by_product_code:
        imputer = method_dict[method]
        imputed_floats = imputer.fit_transform(df[float_cols])
        df[float_cols] = pd.DataFrame(imputed_floats, columns=float_cols)
    else:
        for prod_code in df['product_code'].unique():
            imputer = method_dict[method]
            df.loc[df['product_code']==prod_code, float_cols] = imputer.fit_transform(df.loc[df['product_code']==prod_code, float_cols])

    return df


############################
# Categorical or numerical for attrs 2&3
############################

def assign_attr_types(df:pd.DataFrame, attr_types:dict):
    for value in attr_types.values():
        assert value=='str' or value=='float64', f'"{value}" type is not supported. Only accepted values are "str" and "float64" for categorical and continous variables respectively'
    return df.astype(attr_types)


############################
# One-hot encode
############################

def one_hot_encode(df:pd.DataFrame):
    cat_cols = [f for f in df.columns if df[f].dtype == object]
    cat_cols = list(set(cat_cols) - set(['product_code']))
    for col in cat_cols:
        tempdf = pd.get_dummies(df[col], prefix = col)
        df = pd.merge(left = df, right = tempdf, left_index = True, right_index = True)
    df = df.drop(columns=cat_cols)
    df = df.drop(columns=['attribute_0_material_5']) #prevent dummy variable trap
    return df


############################
# Create basic features
############################

def add_NA_flags(df:pd.DataFrame, cols:list):

    for col in cols:
        df[col+'_missing'] = df[col].isna().apply(lambda x: int(x))
    return df

def add_logic_features(df:pd.DataFrame, logic_dict:dict):

    if logic_dict['use'] == False:
        return df
        
    df['area_attr2_attr_3'] = df['attribute_2'] * df['attribute_3']
    return df

def add_loading_features(df:pd.DataFrame, loading_dict:dict):

    if loading_dict['use'] == False:
        return df

    df['loading_log'] = df['loading'].apply(lambda x: np.log(x))
    df['loading_rank'] = df['loading'].rank()

    return df

############################
# Add polynomial and spline features
############################

def add_poly(df:pd.DataFrame, poly_dict:dict):

    if poly_dict['use'] == False:
        return df

    # Get cols
    continous_cols = [f for f in df.columns if df[f].dtype != object and f != 'failure']
    categorical_cols = list(set(df.columns) - set(continous_cols))

    # Create features
    poly = PolynomialFeatures(degree=poly_dict['degree'])
    new_features = poly.fit_transform(df[continous_cols])
    new_features = pd.DataFrame(new_features, columns=poly.get_feature_names_out())
    new_features = new_features.drop(columns=['measurement_3_missing^2', 'measurement_5_missing^2']) #manual for now

    df = pd.concat([df[categorical_cols], new_features], axis=1)
    return df


def add_spline(df:pd.DataFrame, spline_dict:dict):

    if spline_dict['use'] == False:
        return df

    # Get cols
    continous_cols = [f for f in df.columns if df[f].dtype != object and f != 'failure']

    # Create features
    spline = SplineTransformer(degree=spline_dict['n_knots'], n_knots=spline_dict['degree'])
    new_features = spline.fit_transform(df[continous_cols])
    new_features = pd.DataFrame(new_features, columns=spline.get_feature_names_out())

    df = pd.concat([df, new_features], axis=1)
    return df


############################
# Add group features
############################

def add_deviation_feature(df, feature, category):
    
    # temp groupby object
    category_gb = df.groupby(category)[feature]
    
    # create category means and standard deviations for each observation
    category_mean = category_gb.transform(lambda x: x.mean())
    category_std = category_gb.transform(lambda x: x.std())
    
    # compute stds from category mean for each feature value,
    # add to X as new feature
    deviation_feature = (df[feature] - category_mean) / category_std 
    df[feature + '_Dev_' + category] = deviation_feature 
    return df

def add_group_features(df:pd.DataFrame, group_dict:dict):

    if group_dict['use'] == False:
        return df

    # The weird meas thing
    meas_gr1_cols = [f"measurement_{i:d}" for i in list(range(3, 5)) + list(range(9, 17))]
    meas_gr2_cols = [f"measurement_{i:d}" for i in list(range(5, 9))]
    df['meas_gr1_avg'] = np.mean(df[meas_gr1_cols], axis=1)
    df['meas_gr1_std'] = np.std(df[meas_gr1_cols], axis=1)
    df['meas_gr2_avg'] = np.mean(df[meas_gr2_cols], axis=1)
    df['meas17/meas_gr2_avg'] = df['measurement_17'] / df['meas_gr2_avg']

    # Deviations
    df = add_deviation_feature(df, 'measurement_17', 'product_code')
    df = add_deviation_feature(df, 'measurement_17', 'attribute_0')
    df = add_deviation_feature(df, 'loading', 'product_code')
    df = add_deviation_feature(df, 'loading', 'attribute_0')
    df = add_deviation_feature(df, 'measurement_0', 'product_code')
    df = add_deviation_feature(df, 'measurement_0', 'attribute_0')
    df = add_deviation_feature(df, 'measurement_2', 'product_code')
    df = add_deviation_feature(df, 'measurement_2', 'attribute_0')
    df = add_deviation_feature(df, 'measurement_3', 'product_code')
    df = add_deviation_feature(df, 'measurement_3', 'attribute_0')
    df = add_deviation_feature(df, 'measurement_9', 'product_code')
    df = add_deviation_feature(df, 'measurement_9', 'attribute_0')
    df = add_deviation_feature(df, 'meas_gr1_avg', 'product_code')
    df = add_deviation_feature(df, 'meas_gr2_avg', 'attribute_0')

    return df


############################
# Feature scaling
############################

def scale(df:pd.DataFrame, scale_dict:dict):

    if scale_dict['use'] == False:
        return df

    method_dict = {
        'standard_scaler': StandardScaler(),
        'power_transformer': PowerTransformer(),
        'quantile_transformer': QuantileTransformer()}
    
    exceptions = ['failure', 'measurement_3_missing', 'measurement_5_missing']
    continous_cols = [f for f in df.columns if df[f].dtype != object and f not in exceptions]
    categorical_cols = list(set(df.columns) - set(continous_cols))

    scaler = method_dict[scale_dict['method']]
    new_features = scaler.fit_transform(df[continous_cols])
    new_features = pd.DataFrame(new_features, columns=scaler.get_feature_names_out())

    df = pd.concat([df[categorical_cols], new_features], axis=1)

    return df


############################
# Feature selection
############################

# From https://www.kaggle.com/code/heyspaceturtle/feature-selection-is-all-u-need-2
def FisherScore(bt, y_train, predictors):
    """
    Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights
    into churn prediction in the telecommunication sector: A profit driven data mining
    approach. European Journal of Operational Research, 218(1), 211-229.
    """
    
    # Get the unique values of dependent variable
    target_var_val = y_train.unique()
    
    # Calculate FisherScore for each predictor
    predictor_FisherScore = []
    for v in predictors:
        term1 = np.abs(np.mean(bt.loc[y_train == target_var_val[0], v]) - np.mean(bt.loc[y_train == target_var_val[1], v]))
        term2 = np.sqrt(np.var(bt.loc[y_train == target_var_val[0], v]) + np.var(bt.loc[y_train == target_var_val[1], v]))
        predictor_FisherScore.append(term1 / term2)
    return predictor_FisherScore



def stepwise_feature_selection(df:pd.DataFrame, y:pd.Series, seed:int):
    #Use only train set to fit
    df_train = df[df['product_code'].isin(['A', 'B', 'C', 'D', 'E'])]
    product_code = df_train['product_code']
    float_cols = [f for f in df_train.columns if df_train[f].dtype != object]
    cat_cols = list(set(df_train.columns) - set(float_cols))
    df_train = df_train[float_cols]
    
    # Calculate Fisher Score for all variables
    fs = FisherScore(df_train, y, df_train.columns)
    fs_df = pd.DataFrame({"predictor": df_train.columns, "fisherscore":fs})
    fs_df = fs_df.sort_values('fisherscore', ascending=False).reset_index(drop=True)
    
    # # Check how AUC changes when adding more variables sorted by fisher score importance
    var_list = []
    best_auroc = 0
    print('Stepwise selection progress')

    for _, row in tqdm(fs_df.iterrows()):
        temp_var_list = var_list.copy()
        temp_var_list.append(row['predictor'])

        clf = LogisticRegression(max_iter=215, C=1.67, penalty='l1', solver='liblinear', random_state=seed)
        cv_result = cross_validate(clf, df_train[temp_var_list], y, groups=product_code,
                                   scoring='roc_auc', cv=GroupKFold(n_splits=5), verbose=0, return_train_score=True)
        auroc = np.mean(cv_result['test_score'])

        if auroc >= best_auroc + 0.0001:
            best_auroc = auroc
            var_list.append(row['predictor'])

        # print(f'Vars: {temp_var_list}')
        # print(f'auroc: {auroc}')
        # print('-'*50)

    df = pd.concat([df[cat_cols], df[var_list]], axis=1)
    return df


def feature_selection(df:pd.DataFrame, y:pd.Series, steps:dict, stepwise:bool, seed:int):
    '''
    Example steps (more details in feature_selection_utils.py)
    steps = {'Constant_Features': {'frac_constant_values': 0.9},
             'Correlated_Features': {'correlation_threshold': 0.9},
             'Lasso_Remover': {'alpha': 1, 'coef_threshold':1e-05},
             'Mutual_Info_Remover': {'mi_threshold': 0.05},
             'Boruta_Remover': {'max_depth': 10}}  #Random forest features
    '''

    df_train = df[df['product_code'].isin(['A', 'B', 'C', 'D', 'E'])]
    
    FS = FeatureSelector()
    FS.fit(df_train, y, steps)  #Fit only on train set
    FS.transform(df)  #Transform entire dataser

    if stepwise:
        df = stepwise_feature_selection(df, y, seed)

    return df
