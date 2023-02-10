import scipy.stats
import statistics
import numpy as np
from numpy import absolute
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from IPython.display import display
from typing import Tuple


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA


from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import explained_variance_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LinearRegression
from sklearn import metrics
import random
import xgboost
import shap
from sklearn.metrics import mean_squared_error as MSE
import os
from bartpy.sklearnmodel import SklearnModel


base_path = "C:/Users/Christopher/JLUbox/Transkriptanalysen/2 TOPIC MODELING/Analysen/"
sub_folder_processing = "data/processing"
sub_folder_transkripte = "data/transkripte"
sub_folder_ML = "data/0 ML"

def cor(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

def rsquared_dm(y_pred: np.ndarray, dtrain: xgboost.DMatrix) -> Tuple[str, float]:
    #y_pred[y_pred < -1] = -1 + 1e-6
    """ Return R^2 where x and y are array-like."""
    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_true)
    r_value=.03
    return r_value**2

def split_preparation(test_splits, val_splits, df, outcome):

    y = df[[outcome]]  # Outcome auswählen
    for column in df.columns:
        if "249" in column[0:3]: end_col = column # letztes Topic = 249_... auswählen

    df = df.loc[:,:end_col] # Response-Variablen entfernen
    df[outcome] = y # einzelne Response-Variable hinzufügen

    df = df.dropna() # Missings fliegen raus!
    y = y.dropna() # Auch bei y!


    test_kf = RepeatedKFold(n_splits=test_splits, n_repeats=1, random_state=42)
    val_kf = RepeatedKFold(n_splits=val_splits, n_repeats=1, random_state=42)

    for outer_fold in range(0, test_splits): # hinten dran kommt eine Variable für die folds. Darin steht in jedem Fold, wann man valid-set ist.
        df["fold_" + str(outer_fold)]=-1
    columns = df.columns.tolist()

    a_data = df.values

    for outer_fold, (train_index, test_index) in enumerate(test_kf.split(a_data)): #
        a_train, a_test = a_data[train_index], a_data[test_index]
        print(outer_fold)
        inner_fold = 0
        for strain_index, valid_index in val_kf.split(a_train):
            print(inner_fold)
            a_strain, a_valid = a_train[strain_index], a_train[valid_index]
            df_valid = pd.DataFrame(a_valid, columns=columns)
            session_list = df_valid["session"].tolist()
            df.loc[df['session'].isin(session_list), "fold_" + str(outer_fold)] = inner_fold # folds benennen, soweit eine row im valid-set ist
            inner_fold += 1
    df_cv = df.loc[:, "fold_0":]
    df_ml = df.loc[:, :outcome]
    #df_ml.insert(0, column, value)
    return df_ml, df_cv

def find_params_xgb(X_valid, X_strain, y_valid, y_strain, xgb_params, xgb_r2):
    results = []
    max_depths = list(range(2, 4))
    xgb_params_grid["max_depth"] = max_depths



    # pipeline für xgbr
    clf = GridSearchCV(estimator=xgbr,
                       param_grid=xgb_params_grid,
                       scoring='neg_mean_squared_error',
                       verbose=1, n_jobs=-1, cv=5)

        #training model
    results = clf.fit(X_strain, y_strain, eval_set = [(X_valid, y_valid)], verbose=1) #lasso

    print(results.best_params_)

    subsamples = np.linspace(0.1, 1, 2)
    colsample_bytrees = np.linspace(0.1, 0.5, 2)
    colsample_bylevel = np.linspace(0.1, 0.5, 2)

    # merge into full param dicts

    params_dict = results.best_params_
    params_dict = {md: [params_dict[md]] for md in params_dict}
    # für lasso
    params_dict["subsample"] = subsamples
    params_dict["colsample_bytree"] = colsample_bytrees
    params_dict["colsample_bylevel"] = colsample_bylevel

    clf = GridSearchCV(estimator=xgbr,
                       param_grid=params_dict,
                       scoring='neg_mean_squared_error',
                       verbose=1, n_jobs=-1, cv=5)

    results = clf.fit(X_strain, y_strain, eval_set = [(X_valid,y_valid)], verbose=0) #lasso
    print(results.best_params_)

    learning_rates = np.logspace(-3, -0.7, 3)
    params_dict = results.best_params_
    params_dict = {md: [params_dict[md]] for md in params_dict}
    params_dict["learning_rate"]=learning_rates

    clf = GridSearchCV(estimator=xgbr,
                       param_grid=params_dict,
                       scoring='neg_mean_squared_error',
                       verbose=1, n_jobs=-1, cv=5)


    results = clf.fit(X_strain, y_strain, eval_set = [(X_valid,y_valid)], verbose=0)
    print(results.best_params_)


    best_model = results.best_estimator_
    xgb_params.append(clf.best_params_)
    y_pred_valid = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1,)
    xgb_r2.append(rsquared(y_valid, y_pred_valid))

    return xgb_r2, xgb_params

def find_params_rf(X_valid, X_strain, y_valid, y_strain, rf_params, rf_r2):
    mod_rf = GridSearchCV(estimator=rf,
                                param_grid=rf_params_grid,
                                cv=5,
                                n_jobs=-1,
                                verbose=2)
    y_strain = y_strain.reshape(-1,)
    results = mod_rf.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    rf_params.append(mod_rf.best_params_)
    y_pred_valid = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1,)
    rf_r2.append(rsquared(y_valid, y_pred_valid))

    return rf_r2, rf_params

def find_params_lasso(X_valid, X_strain, y_valid, y_strain, lasso_params, lasso_r2):
    results = mod_lasso.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    lasso_params.append(mod_lasso.best_params_)
    y_pred_valid = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1,)
    lasso_r2.append(rsquared(y_valid, y_pred_valid))

    return lasso_r2, lasso_params

def find_params_bart(X_valid, X_strain, y_valid, y_strain, bart_params, bart_r2):
    mod_bart = GridSearchCV(estimator=bart,
                          param_grid=bart_params_grid,
                          cv=5,
                          n_jobs=-1,
                          verbose=2)
    y_strain = y_strain.reshape(-1,)
    results = mod_bart.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    bart_params.append(mod_lasso.best_params_)

    y_pred_valid = bart.predict(X_valid)
    y_valid = y_valid.reshape(-1, )
    bart_r2.append(rsquared(y_valid, y_pred_valid))
    return bart_params, bart_r2


def cv_with_arrays(df_ml_to_array, df_cv_to_array, val_splits):
    array_ml, array_cv = df_ml_to_array.values, df_cv_to_array.values
    xgb_r2, xgb_params, lasso_r2, lasso_params, rf_r2, rf_params = [], [], [], [], [], []
    for col in range(array_cv.shape[1]): # Jede spalte durchgehen
        for inner_fold in range(val_splits): # jede Zeile durchgehen
            array_valid = array_ml[array_cv[:, col]==inner_fold] # X_valid erstellen
            array_strain = array_ml[(array_cv[:, col]!=inner_fold) & (array_cv[:, col]!=-1)] # X strain ist ungleich x_valid und ungleich x_test
            X_valid = array_valid[:, :-1]
            X_strain = array_strain[:, :-1]
            y_valid = array_valid[:, [-1]]
            y_strain = array_strain[:, [-1]]
            #xgb_r2, xgb_params = find_params_xgb(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain,
            #                                     xgb_params=xgb_params,xgb_r2=xgb_r2)

            # lasso_r2, lasso_params = find_params_lasso(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain,
            #                                     lasso_params=lasso_params,lasso_r2=lasso_r2) # Lasso war nicht gut!

            #rf_r2, rf_params = find_params_rf(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain,
            #                                     rf_params=rf_params,rf_r2=rf_r2)


    return rf_r2, rf_params, xgb_r2, xgb_params, lasso_r2, lasso_params, X_strain, X_valid, y_strain, y_valid



# xgboost model
xgb_params_grid = {'max_depth': [2],
           'learning_rate': [0.03],
           'n_estimators': [2000],
           'subsample': [0.4],
           'colsample_bylevel': [0.1],
           'colsample_bytree': [0.1],
           'early_stopping_rounds': [500]}

xgbr = xgboost.XGBRegressor(seed = 20, objective='reg:squarederror',
                            booster='gbtree')
# Lasso Model
pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ('model', Lasso())
])

mod_lasso=GridSearchCV(pipeline, {"model__alpha":np.arange(0.01, 0.5, 0.005)}, cv=5, scoring="neg_mean_squared_error",
                        verbose=3, n_jobs=-1)
# Random Forest Model
rf_params_grid = {
    'bootstrap': [True],
    'max_depth': [2, 5, 8],
    'max_features': [2, 5, 10],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 20, 50],
    'n_estimators': [500]
}

rf = RandomForestRegressor()

# BART Model
bart_params_grid = {
    'n_trees': [200],
    'alpha': [0.90],
    'beta': [3],
}

bart = SklearnModel()

path = os.path.join(base_path,sub_folder_ML)
os.chdir(path)
print(path)
df = pd.read_excel('topic_document_matrix_sum.xlsx', index_col=0)

test_sets, val_sets = 10, 5
outcome = "hscl" # Alternativ "srs_ges"


df_ml, df_nested_cv = split_preparation(test_splits=test_sets, val_splits=val_sets, df=df, outcome=outcome) # Alternativ "srs_ges"
df_ml = df_ml.iloc[:,1:] # erste Spalte löschen (session-Variable ist nicht ml-geeignet)

rf_r2, rf_params, xgb_r2, xgb_params, lasso_r2, lasso_params, X_strain, X_valid, y_strain, y_valid = cv_with_arrays(df_ml_to_array=df_ml, df_cv_to_array=df_nested_cv, val_splits=val_sets)
np.median(lasso_r2)

df.to_excel("saved_folds.xlsx", index=False)



RANDOMSTATE = 42
BOOST_ROUNDS=50000   # we use early stopping so make this arbitrarily high
EARLY_STOPPING_ROUNDS=100 # stop if no improvement after 100 rounds



def my_cv(df, predictors, response, rkf, regressor, verbose=False):
    """Roll our own CV
    train each kfold with early stopping
    return average metric, sd over kfolds, average best round"""
    metrics_list_nrmse, r2_list = [], []
    best_iterations = []

    for train_fold, cv_fold in rkf.split(df):
        fold_X_train=df[predictors].values[train_fold]
        fold_y_train=df[response].values[train_fold]
        fold_X_test=df[predictors].values[cv_fold]
        fold_y_test=df[response].values[cv_fold]
        regressor.fit(fold_X_train, fold_y_train,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      eval_set=[(fold_X_test, fold_y_test)],
                      verbose=verbose
                     )
        y_pred_test=regressor.predict(fold_X_test)
        metrics_list_nrmse.append(np.sqrt(metrics.mean_squared_error(fold_y_test, y_pred_test, squared=False)/statistics.stdev(fold_y_test)))
        r2_list.append(rsquared(fold_y_test, y_pred_test))
        best_iterations.append(regressor.best_iteration)
    return np.average(metrics_list_nrmse), np.std(metrics_list_nrmse), np.average(r2_list), np.average(best_iterations)

def cv_over_param_dict(df, param_dict, predictors, response, rkf, verbose=False):
    """given a list of dictionaries of xgb params
    run my_cv on params, store result in array
    return results
    """
    results = []
    for i, d in enumerate(param_dict):
        xgb = xgboost.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=BOOST_ROUNDS,
            random_state=RANDOMSTATE,
            verbosity=1,
            n_jobs=-1,
            booster='gbtree',
            **d
        )

        metric_nrmse, metric_std, metric_r2, best_iteration = my_cv(df, predictors, response, rkf, xgb, verbose=False)
        results.append([metric_nrmse, metric_std, metric_r2, best_iteration, d])
    results_df = pd.DataFrame(results, columns=['nrmse', 'std', 'metric_r2', 'best_iter', 'param_dict']).sort_values('nrmse')
    display(results_df.head())

    best_params = results_df.iloc[0]['param_dict']
    return best_params, results_df


df_imp = pd.read_excel(r'C:\Users\Christopher\Documents\PsyRes\Bern-Psychotherapeutenstudie\Output\Veränderung\imp_mf.xlsx')
# categorical variables: Jahrgang, WBS, Familienstand
list_cat = []
for cat_var in ["Jahrgang", "WBS", "Familienstand"]:
    if cat_var in df_imp.columns.tolist():
        list_cat.append(cat_var)

for cat in list_cat:
    df_imp = encode_and_bind(df_imp, cat)
df_imp.to_excel(r'C:\Users\Christopher\Documents\PsyRes\Bern-Psychotherapeutenstudie\Output\Veränderung\reg_r.xlsx')


df_imp = df_imp.drop('TH_code', 1)

# Seperate between Outcome and features
#Achtung!!!! Anfang anpassen
X = np.array(df_imp.loc[:, "WBS1":"Bindung"])
y = np.array(df_imp["GAS"])
columns = list(df_imp.loc[:, "WBS1":"Bindung"])

# choose lasso predictors for xgboost
df_sel = pd.read_excel(r'C:\Users\Christopher\Documents\PsyRes\Bern-Psychotherapeutenstudie\Output\percent_change\lasso_shap\shap_lasso_fin.xlsx')
feature_list = []
for i in range(0, len(df_sel)):
    if df_sel["percent_shap_value"][i] > 0.1:
        feature_list.append(df_sel["Feature"][i])
#Xgboost nur für Lasso-Features ACHTUNG!!!!
X = np.array(df_imp[feature_list])
y = np.array(df_imp["percent_change"])
columns = list(df_imp[feature_list])



# xgboost# Seperate between Outcome and features für pca
# #Achtung!!!! Anfang anpassen
X = np.array(df_imp.loc[:, "WBS1":"Bindung"])
y = np.array(df_imp["GAS"])
columns = list(df_imp.loc[:, "WBS1":"Bindung"])

# initial hyperparams
current_params = {
    'max_depth': 2,
    'colsample_bytree': 0.3,
    'colsample_bylevel': 0.3,
    'subsample': 0.4,
    'learning_rate': 0.03,
}

df = df_imp
predictors = feature_list
response = "percent_change"
rkf = RepeatedKFold(n_splits=20, n_repeats=5)
##################################################
# round 1: tune depth , max_leaves
##################################################
max_depths = list(range(2,4))
# max_leavess = [1, 3, 10, 30, 100] #  doesn't matter
# grid_search_dicts = [dict(zip(['max_depth', 'max_leaves'], [a, b]))
#                      for a,b in product(max_depths, max_leavess)]
grid_search_dicts = [{'max_depth': md} for md in max_depths]
# merge into full param dicts
full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]

# cv and get best params
current_params, results_df = cv_over_param_dict(df, full_search_dicts, predictors, response, rkf, verbose=False)

##################################################
# round 2: tune subsample, colsample_bytree, colsample_bylevel
##################################################
subsamples = np.linspace(0.1, 0.4, 3)
colsample_bytrees = np.linspace(0.1, 0.3, 3)
colsample_bylevel = np.linspace(0.1, 0.3, 3)
# narrower search
# subsamples = np.linspace(0.25, 0.35, 11)
# colsample_bytrees = np.linspace(0.1, 0.3, 21)
# colsample_bylevel = np.linspace(0.1, 0.3, 21)

grid_search_dicts = [dict(zip(['subsample', 'colsample_bytree', 'colsample_bylevel'], [a, b, c]))
                     for a,b,c in itertools.product(subsamples, colsample_bytrees, colsample_bylevel)]
# merge into full param dicts
full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]
# cv and get best params
current_params, results_df = cv_over_param_dict(df, full_search_dicts, predictors, response, rkf, verbose=False)

##################################################
# round 3: learning rate
##################################################
learning_rates = np.logspace(-3, -1, 5)
grid_search_dicts = [{'learning_rate': lr} for lr in learning_rates]
# merge into full param dicts
full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]

# cv and get best params
current_params, results_df = cv_over_param_dict(df, full_search_dicts, predictors, response, rkf, verbose=False)


# Parameter tuning lasso, set equal to pre-selection
params = {'max_depth': [3],
           'learning_rate': [0.1],
           'n_estimators': [2000],
           'subsample': [0.25],
           'colsample_bylevel': [0.1],
           'colsample_bytree': [0.1]}

# parameter tuning pca
params = {'xgb__max_depth': [2],
           'xgb__learning_rate': [0.03],
           'xgb__n_estimators': [2000],
           'xgb__subsample': [0.4],
           'xgb__colsample_bylevel': [0.1],
           'xgb__colsample_bytree': [0.1]}
xgbr = xgboost.XGBRegressor(seed = 20, objective='reg:squarederror',
                            booster='gbtree')
# nur für PCA
pipeline = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('pca', PCA(50)),
    ('xgb', xgbr)
])

list_shap_values = list()
list_best_params = []
l_rmse, l_mae, l_r2, l_nrmse, l_r2_train, l_nrmse_train = [], [], [], [], [], []
z=0
shap_values = None
rkf = RepeatedKFold(n_splits=20, n_repeats=5, random_state=42)
grid_search = False

for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_strain, X_valid, y_strain, y_valid = train_test_split(X_train, y_train, test_size=.1)

    X_train = pd.DataFrame(X_train, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)
    X_strain = pd.DataFrame(X_strain, columns=columns)
    X_valid = pd.DataFrame(X_valid, columns=columns)

    if not grid_search:
        clf = GridSearchCV(estimator=xgbr,
                           param_grid=params,
                           scoring='explained_variance',
                           verbose=1, n_jobs=-1, cv=5)
        results = clf.fit(X_strain, y_strain, early_stopping_rounds=500, eval_set=[(X_valid, y_valid)],
                          verbose=0)  # lasso
        #results = clf.fit(X_strain, y_strain, xgb__early_stopping_rounds = 100, xgb__eval_set = [(pip.transform(X_valid), y_valid)], xgb__verbose=0) #pca
    else:
        max_depths = list(range(2, 5))

        params["max_depth"]=max_depths # für lasso
        #params["xgb__max_depth"]=max_depths # für pca



        # pipeline für pca, xgbr für lasso
        clf = GridSearchCV(estimator=xgbr,
                       param_grid=params,
                       scoring='neg_mean_squared_error',
                       verbose=1, n_jobs=-1, cv=5)

        # create short pipeline only with standardscaler and pca
        #pipeline_short = pipeline[:-1]
        #pip = pipeline_short.fit(X_strain)
        #training model
        results = clf.fit(X_strain, y_strain, early_stopping_rounds = 500, eval_set = [(X_valid, y_valid)], verbose=0) #lasso
        #results = clf.fit(X_strain, y_strain, xgb__early_stopping_rounds = 100, xgb__eval_set = [(pip.transform(X_valid), y_valid)], xgb__verbose=0) #pca

        print(results.best_params_)
        print(abs(results.best_score_)**0.5)
        subsamples = np.linspace(0.1, 0.5, 4)
        colsample_bytrees = np.linspace(0.1, 0.3, 3)
        colsample_bylevel = np.linspace(0.1, 0.3, 3)

        # merge into full param dicts
        params_dict = results.best_params_
        params_dict = {md: [params_dict[md]] for md in params_dict}
        # für lasso
        params_dict["subsample"] = subsamples
        params_dict["colsample_bytree"] = colsample_bytrees
        params_dict["colsample_bylevel"] = colsample_bylevel

        # für pca
        #params_dict["xgb__subsample"] = subsamples
        #params_dict["xgb__colsample_bytree"] = colsample_bytrees
        #params_dict["xgb__colsample_bylevel"] = colsample_bylevel
        # pipeline für pca, xgbr für lasso
        clf = GridSearchCV(estimator=xgbr,
                       param_grid=params_dict,
                       scoring='neg_mean_squared_error',
                       verbose=1, n_jobs=-1, cv=5)

        #training model, early stopping und eval_set geht nicht mit pca
        results = clf.fit(X_strain, y_strain, early_stopping_rounds = 100, eval_set = [(X_valid,y_valid)], verbose=0) #lasso
        #results = clf.fit(X_strain, y_strain, xgb__early_stopping_rounds = 100, xgb__eval_set = [(pip.transform(X_valid), y_valid)], xgb__verbose=0) #pca
        print(results.best_params_)
        print(abs(results.best_score_)**0.5)

        learning_rates = np.logspace(-3, -0.7, 5)
        params_dict = results.best_params_
        params_dict = {md: [params_dict[md]] for md in params_dict}
        params_dict["learning_rate"]=learning_rates # für lasso
        #params_dict["xgb__learning_rate"] = learning_rates # für pca


        # pipeline für pca, xgbr für lasso
        clf = GridSearchCV(estimator=xgbr,
                       param_grid=params_dict,
                       scoring='neg_mean_squared_error',
                       verbose=1, n_jobs=-1, cv=5)

        #training model, early stopping und eval_set geht nicht mit pca
        results = clf.fit(X_strain, y_strain, early_stopping_rounds = 100, eval_set = [(X_valid,y_valid)], verbose=0) #lasso
        #results = clf.fit(X_strain, y_strain, xgb__early_stopping_rounds = 100, xgb__eval_set = [(pip.transform(X_valid), y_valid)], xgb__verbose=0) #pca
        print(results.best_params_)
        print(abs(results.best_score_)**0.5)

    best_model = results.best_estimator_
    list_best_params.append(clf.best_params_)
    y_pred = best_model.predict(X_test)

    #train
    y_pred_train = best_model.predict(X_train)
    print(rsquared(y_test, y_pred))
    print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(metrics.mean_squared_error(y_test, y_pred, squared=False)/statistics.stdev(y_test))
    l_r2_train.append(rsquared(y_train, y_pred_train))
    l_nrmse_train.append(metrics.mean_squared_error(y_train, y_pred_train, squared=False)/statistics.stdev(y_train))

    l_rmse.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    l_r2.append(rsquared(y_test, y_pred))
    l_mae.append(metrics.mean_absolute_error(y_test, y_pred))
    l_nrmse.append(metrics.mean_squared_error(y_test, y_pred, squared=False)/statistics.stdev(y_test))

    #explaining model lasso
    explainer_xgb = shap.TreeExplainer(best_model, X_test)
    shap_values = explainer_xgb(X_test, check_additivity=False)

    #explaining model pca
    #explainer_xgb = shap.TreeExplainer(best_model.named_steps["xgb"], X_test)
    #shap_values = explainer_xgb(best_model[:-1].transform(X_test), check_additivity=False)
    #for each iteration we save the test_set index and the shap_values
    list_shap_values.append(shap_values)
    z += 1

    print(z)


df_accuracy = pd.DataFrame({"MAE": l_mae, "r2": l_r2, "rmse": l_rmse, "nrmse": l_nrmse})
df_accuracy.to_excel(r'C:\Users\Christopher\Documents\PsyRes\Bern-Psychotherapeutenstudie\Output\percent_change\xgboost_lasso_2\Accuracy_xgb_prelasso.xlsx')

np.abs(df_accuracy["r2"]).median()
np.abs(df_accuracy["r2"]).std()
np.abs(df_accuracy["r2"]).mean()

np.abs(df_accuracy["MAE"]).mean()
np.abs(df_accuracy["MAE"]).median()
np.abs(df_accuracy["MAE"]).std()
np.abs(df_accuracy["nrmse"]).mean()
np.abs(df_accuracy["rmse"]).std()
np.abs(df_accuracy["nrmse"]).median()

sh_values, bs_values, sh_data = None, None, None
sh_values = list_shap_values[0].values
bs_values = list_shap_values[0].base_values
sh_data = list_shap_values[0].data
for i in range(1, len(list_shap_values)):
    sh_values = np.concatenate((sh_values,np.array(list_shap_values[i].values)), axis=0)
    bs_values = np.concatenate((bs_values,np.array(list_shap_values[i].base_values)), axis=0)
    sh_data = np.concatenate((sh_data, np.array(list_shap_values[i].data)), axis=0)

shap_values_agg = shap.Explanation(values=sh_values,
                                   base_values=bs_values, data=sh_data,
                                   feature_names=list_shap_values[0].feature_names)

shap.plots.beeswarm(shap_values_agg, plot_size=(25, 22), max_display=35, show=False)
plt.savefig(r'C:\Users\Christopher\Documents\PsyRes\Bern-Psychotherapeutenstudie\Output\percent_change\xgboost_lasso_2\summary_xgboost_prelasso.png')


df_sh_values = pd.DataFrame(sh_values)
df_sh_values.to_excel(r'C:\Users\Christopher\Documents\PsyRes\Bern-Psychotherapeutenstudie\Output\percent_change\xgboost_lasso_2\sh_values.xlsx')
df_bs_values = pd.DataFrame(bs_values)
df_bs_values.to_excel(r'C:\Users\Christopher\Documents\PsyRes\Bern-Psychotherapeutenstudie\Output\percent_change\xgboost_lasso_2\bs_values.xlsx')
df_sh_data = pd.DataFrame(sh_data)
df_sh_data.to_excel(r'C:\Users\Christopher\Documents\PsyRes\Bern-Psychotherapeutenstudie\Output\percent_change\xgboost_lasso_2\sh_data.xlsx')
df_feature_names = pd.DataFrame(list_shap_values[0].feature_names)
df_feature_names.to_excel(r'C:\Users\Christopher\Documents\PsyRes\Bern-Psychotherapeutenstudie\Output\percent_change\xgboost_lasso_2\feature_names.xlsx')



global_shap_values = np.abs(shap_values_agg.values).mean(0)
df_shap_values = pd.DataFrame(global_shap_values.reshape(-1, len(global_shap_values)), columns = columns)


df_shap_values_new = pd.DataFrame({"Feature": df_shap_values.columns.tolist(), "SHAP-value": df_shap_values.iloc[0].tolist()})
df_shap_values_new["SHAP-value"].sum()
df_shap_values_new["percent_shap_value"] = df_shap_values_new["SHAP-value"] / df_shap_values_new["SHAP-value"].sum() * 100
df_shap_values_new.to_excel(r'C:\Users\Christopher\Documents\PsyRes\Bern-Psychotherapeutenstudie\Output\percent_change\xgboost_lasso_2\shap_values_xgb_prelasso.xlsx')

