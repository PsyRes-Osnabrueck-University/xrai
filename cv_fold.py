import json
import scipy.stats
import statistics
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, GroupKFold
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from merf import MERF
import gpboost as gpb


from featurewiz import FeatureWiz
from sklearn import metrics
import xgboost

from sklearn.metrics import make_scorer

def cor(x, y):
    """ Return R where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value

r_scorer = make_scorer(cor, greater_is_better=True)
# LASSO
lasso_pipeline = Pipeline([("scaler", StandardScaler()), ('model', Lasso(random_state=42))])

# ELASTIC NET
e_net_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ('model', ElasticNet(random_state=42))
])

ratios = np.arange(0.001, 0.3, 0.003)
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 5, 10]

mod_e_net = GridSearchCV(e_net_pipeline, {"model__alpha": alphas, "model__l1_ratio": ratios}, cv=5, scoring="r2", verbose=0, n_jobs=-1)



# Random Forest Model
rf_params_grid = {
    'bootstrap': [True],
    'max_depth': [15],
    'max_features': [10, 20],
    'min_samples_leaf': [2,4],
    'min_samples_split': [2, 4],
    'n_estimators': [1000]
}

rf = RandomForestRegressor(random_state=42)

mod_rf = GridSearchCV(estimator=rf, param_grid=rf_params_grid, scoring="r2", cv=5, n_jobs=-1, verbose=0)

# mixed effects random forest (MERF)
merf = MERF(max_iterations=5)

# SVM

svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Auch hier müssen wir zuerst skalieren
    ('model', SVR())
])


svr_params_grid = {
    'model__kernel': ['rbf'],
    'model__C': [1, 2, 3],  # hatte 1 oder 2 als Optimum
    'model__degree': [2, 3],
    'model__coef0': [0.000001, 0.000005, 0.00001],  # hatte 0.001 als Optimum
    'model__gamma': ['auto', 'scale']}

mod_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_params_grid, scoring="r2", cv=5, n_jobs=-1, verbose=0)

# GPBOOST
likelihood = "gaussian"


gpb_param_grid = {'learning_rate': [1,0.1,0.01],
              'min_data_in_leaf': [10,100],
              'max_depth': [2, 5],
              'lambda_l2': [0,1,10]}
gpb_other_params = {'num_leaves': 2**10, 'verbose': 0}

# Ensemble:
ensemble_params_grid = {
    'model__kernel': ['rbf'],
    'model__C': [1, 2, 3, 5, 6, 7, 8, 9, 10],  # hatte 1 oder 2 als Optimum
    'model__degree': [2, 3],
    'model__coef0': [1e-20, 1e-15, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  # hatte 0.001 als Optimum
    'model__gamma': ['auto', 'scale']}

mod_meta = GridSearchCV(estimator=svr_pipeline, param_grid=ensemble_params_grid, cv=5,
                            scoring="r2", verbose=0, n_jobs=-1)

fwiz = FeatureWiz(corr_limit=0.8, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=0)

def scorer_fun(y_true, y_pred):
    return -metrics.mean_squared_error(y_true, y_pred, squared=True)


def find_params_xgb(X_valid, X_strain, y_valid, y_strain, params, r, nrmse, classed=False, group_strain=[]):

    xgb_params_grid = {'max_depth': [2], 'learning_rate': [0.121], 'n_estimators': [100, 500], 'subsample': [0.5],
                       'colsample_bylevel': [0.275], 'colsample_bytree': [0.275]}

    xgbr = xgboost.XGBRegressor(seed=20, objective='reg:squarederror', booster='gbtree')
    max_depths = list(range(2, 10))
    xgb_params_grid["max_depth"] = max_depths

    # pipeline für xgbr
    if not classed: clf = GridSearchCV(estimator=xgbr, param_grid=xgb_params_grid, scoring="r2", verbose=0, n_jobs=-1, cv=5)
    else:
        gkf = list(GroupKFold(n_splits=5).split(X_strain, y_strain, group_strain))
        clf = GridSearchCV(estimator=xgbr, param_grid=xgb_params_grid, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)
    # training model
    results = clf.fit(X_strain, y_strain, verbose=10)


    subsamples = [0.1, 0.5, 1]
    colsample_bytrees = np.linspace(0.05, 0.5, 3)
    colsample_bylevel = np.linspace(0.05, 0.5, 3)

    print(results.best_params_)
    # merge into full param dicts

    params_dict = results.best_params_
    params_dict = {md: [params_dict[md]] for md in params_dict}
    # für lasso
    params_dict["subsample"] = subsamples
    params_dict["colsample_bytree"] = colsample_bytrees
    params_dict["colsample_bylevel"] = colsample_bylevel

    if not classed: clf = GridSearchCV(estimator=xgbr, param_grid=params_dict, scoring="r2", verbose=0, n_jobs=-1, cv=5)
    else:
        gkf = list(GroupKFold(n_splits=5).split(X_strain, y_strain, group_strain))
        clf = GridSearchCV(estimator=xgbr, param_grid=params_dict, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

    results = clf.fit(X_strain, y_strain, verbose=10)
    print(results.best_params_)

    learning_rates = np.logspace(-5, -0.1, 7)
    params_dict = results.best_params_
    params_dict = {md: [params_dict[md]] for md in params_dict}
    params_dict["learning_rate"] = learning_rates

    if not classed: clf = GridSearchCV(estimator=xgbr, param_grid=params_dict, scoring="r2", verbose=0, n_jobs=-1, cv=5)
    else:
        gkf = list(GroupKFold(n_splits=5).split(X_strain, y_strain, group_strain))
        clf = GridSearchCV(estimator=xgbr, param_grid=params_dict, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

    results = clf.fit(X_strain, y_strain, verbose=10)
    print(results.best_params_)

    best_model = clf.best_estimator_

    xgb_pred = best_model.predict(X_valid)
    xgb_pred = xgb_pred.reshape(-1,)
    y_valid = y_valid.reshape(-1, )
    if "xgb" in params:
        params["xgb"].append(results.best_params_)
        r["xgb"].append(cor(y_valid, xgb_pred))
        nrmse["xgb"].append(metrics.mean_squared_error(y_valid, xgb_pred, squared=False) / statistics.stdev(y_valid))
    else:
        params["xgb"] = [results.best_params_]
        r["xgb"] = [cor(y_valid, xgb_pred)]
        nrmse["xgb"] = [metrics.mean_squared_error(y_valid, xgb_pred, squared=False) / statistics.stdev(y_valid)]
    print("XGB_nrmse: " + str(nrmse["xgb"][-1]))
    print("XGB_r: " + str(r["xgb"][-1]))
    return params, r, nrmse


def find_params_rf(X_valid, X_strain, y_valid, y_strain, params, r, nrmse, classed=False, group_strain=[]):

    y_strain = y_strain.reshape(-1, )
    if not classed: mod_rf = GridSearchCV(estimator=rf, param_grid=rf_params_grid, scoring="r2", cv=5, n_jobs=-1, verbose=0)
    else:
        gkf = list(GroupKFold(n_splits=5).split(X_strain, y_strain, group_strain))
        mod_rf = GridSearchCV(estimator=rf, param_grid=rf_params_grid, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

    results = mod_rf.fit(X_strain, y_strain)
    best_model = results.best_estimator_

    rf_pred = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1, )
    if "rf" in params:
        params["rf"].append(results.best_params_)
        r["rf"].append(cor(y_valid, rf_pred))
        nrmse["rf"].append(metrics.mean_squared_error(y_valid, rf_pred, squared=False) / statistics.stdev(y_valid))
    else:
        params["rf"] = [results.best_params_]
        r["rf"] = [cor(y_valid, rf_pred)]
        nrmse["rf"] = [metrics.mean_squared_error(y_valid, rf_pred, squared=False) / statistics.stdev(y_valid)]
    print("RF_nrmse: " + str(nrmse["rf"][-1]))
    print("RF_r: " + str(r["rf"][-1]))
    return params, r, nrmse

def find_params_merf(X_valid, X_strain, y_valid, y_strain, group_strain, group_valid, random_effects, r, nrmse):
    z = np.array([1] * len(X_strain)).reshape(-1,1)
    z = np.hstack([z, X_strain[:, random_effects]])
    y_strain = y_strain.reshape(-1,)
    group_strain = group_strain.reset_index(drop=True)
    group_valid = group_valid.reset_index(drop=True)
    merf.fit(X = np.delete(X_strain, random_effects, axis=1), Z=z, clusters=group_strain, y=y_strain)

    z = np.array([1] * len(X_valid)).reshape(-1,1)
    z = np.hstack([z, X_valid[:, random_effects]])
    merf_pred = merf.predict(X=np.delete(X_valid, random_effects, axis=1),Z=z, clusters=group_valid)
    y_valid = y_valid.reshape(-1, )
    if "merf" in r:
        r["merf"].append(cor(y_valid, merf_pred))
        nrmse["merf"].append(metrics.mean_squared_error(y_valid, merf_pred, squared=False) / statistics.stdev(y_valid))
    else:
        r["merf"] = [cor(y_valid, merf_pred)]
        nrmse["merf"] = [metrics.mean_squared_error(y_valid, merf_pred, squared=False) / statistics.stdev(y_valid)]
    print("merf_nrmse: " + str(nrmse["merf"][-1]))
    print("merf_r: " + str(r["merf"][-1]))
    return r, nrmse
def find_params_lasso(X_valid, X_strain, y_valid, y_strain, params, r, nrmse, classed=False, group_strain=[]):

    if not classed: mod_lasso = GridSearchCV(lasso_pipeline, {"model__alpha": np.arange(0.02, 2.0, 0.005)}, cv=5, scoring="r2", verbose=0, n_jobs=-1)
    else:
        gkf = list(GroupKFold(n_splits=5).split(X_strain, y_strain, group_strain))
        mod_lasso = GridSearchCV(lasso_pipeline, {"model__alpha": np.arange(0.02, 2.0, 0.005)}, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

    results = mod_lasso.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    lasso_pred = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1, )

    if "lasso" in params:
        params["lasso"].append(results.best_params_)
        r["lasso"].append(cor(y_valid, lasso_pred))
        nrmse["lasso"].append(metrics.mean_squared_error(y_valid, lasso_pred, squared=False) / statistics.stdev(y_valid))
    else:
        params["lasso"] = [results.best_params_]
        r["lasso"] = [cor(y_valid, lasso_pred)]
        nrmse["lasso"] = [metrics.mean_squared_error(y_valid, lasso_pred, squared=False) / statistics.stdev(y_valid)]
    print("Lasso_nrmse: " + str(nrmse["lasso"][-1]))
    print("Lasso_r: " + str(r["lasso"][-1]))
    return params, r, nrmse


def find_params_e_net(X_valid, X_strain, y_valid, y_strain, params, r, nrmse, classed=False, group_strain=[]):


    if not classed: mod_e_net = GridSearchCV(e_net_pipeline, {"model__alpha": alphas, "model__l1_ratio": ratios}, cv=5, scoring="r2", verbose=0, n_jobs=-1)
    else:
        gkf = list(GroupKFold(n_splits=5).split(X_strain, y_strain, group_strain))
        mod_e_net = GridSearchCV(e_net_pipeline, {"model__alpha": alphas, "model__l1_ratio": ratios}, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

    results = mod_e_net.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    e_net_pred = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1, )

    if "e_net" in params:
        params["e_net"].append(results.best_params_)
        r["e_net"].append(cor(y_valid, e_net_pred))
        nrmse["e_net"].append(metrics.mean_squared_error(y_valid, e_net_pred, squared=False) / statistics.stdev(y_valid))
    else:
        params["e_net"] = [results.best_params_]
        r["e_net"] = [cor(y_valid, e_net_pred)]
        nrmse["e_net"] = [metrics.mean_squared_error(y_valid, e_net_pred, squared=False) / statistics.stdev(y_valid)]
    print("E_net_nrmse: " + str(nrmse["e_net"][-1]))
    print("E_net_r: " + str(r["e_net"][-1]))
    print("E_net_params: " + str(params["e_net"][-1]))
    return params, r, nrmse


def find_params_gpb(X_valid, X_strain, y_valid, y_strain, group_strain, group_valid, random_effects, params, r, nrmse):
    if random_effects:
        gp_model = gpb.GPModel(group_data=group_strain, group_rand_coef_data=X_strain[:, random_effects],
                               ind_effect_group_rand_coef=[1] * len(random_effects), likelihood="gaussian")
    else:
        gp_model = gpb.GPModel(group_data=group_strain, likelihood="gaussian")
    y_strain = y_strain.reshape(-1, )
    data_train = gpb.Dataset(data=np.delete(X_strain, random_effects, axis=1), label=y_strain)
    opt_params = gpb.grid_search_tune_parameters(param_grid=gpb_param_grid, params=gpb_other_params,
                                                 num_try_random=None, nfold=5, seed=1000, metric="rmse",
                                                 train_set=data_train, gp_model=gp_model,
                                                 use_gp_model_for_validation=True, verbose_eval=0, early_stopping_rounds=10,
                                                 num_boost_round=200)

    print(opt_params)
    bst = gpb.train(params=opt_params['best_params'], train_set=data_train,
                    gp_model=gp_model, num_boost_round=200)
    if random_effects:
        pred = bst.predict(data=X_valid, group_data_pred=group_valid,group_rand_coef_data_pred=X_strain[:, random_effects],
                       predict_var=True, pred_latent=False)
    else:
        pred = bst.predict(data=X_valid, group_data_pred=group_valid,
                       predict_var=True, pred_latent=False)
    pred_gpb = pred["response_mean"].reshape(-1,)
    y_valid = y_valid.reshape(-1, )

    if "gpb" in params:
        params["gpb"].append(opt_params['best_params'])
        r["gpb"].append(cor(y_valid, pred_gpb))
        nrmse["gpb"].append(metrics.mean_squared_error(y_valid, pred_gpb, squared=False) / statistics.stdev(y_valid))
    else:
        params["gpb"] = [opt_params['best_params']]
        r["gpb"] = [cor(y_valid, pred_gpb)]
        nrmse["gpb"] = [metrics.mean_squared_error(y_valid, pred_gpb, squared=False) / statistics.stdev(y_valid)]
    print("GPB_nrmse: " + str(nrmse["gpb"][-1]))
    print("GPB_r: " + str(r["gpb"][-1]))
    print("GPB_params: " + str(params["gpb"][-1]))
    return params, r, nrmse

def find_params_svr(X_valid, X_strain, y_valid, y_strain, params, r, nrmse, classed=False, group_strain=[]):

    if not classed: mod_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_params_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0)
    else:
        gkf = list(GroupKFold(n_splits=5).split(X_strain, y_strain, group_strain))
        mod_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_params_grid, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

    y_strain = y_strain.reshape(-1, )

    results = mod_svr.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    svr_pred = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1, )
    svr_pred = svr_pred.reshape(-1,)

    if "svr" in params:
        params["svr"].append(results.best_params_)
        r["svr"].append(cor(y_valid, svr_pred))
        nrmse["svr"].append(metrics.mean_squared_error(y_valid, svr_pred, squared=False) / statistics.stdev(y_valid))
    else:
        params["svr"] = [results.best_params_]
        r["svr"] = [cor(y_valid, svr_pred)]
        nrmse["svr"] = [metrics.mean_squared_error(y_valid, svr_pred, squared=False) / statistics.stdev(y_valid)]
    print("SVR_nrmse: " + str(nrmse["svr"][-1]))
    print("SVR_r: " + str(r["svr"][-1]))
    print("SVR_params: " + str(params["svr"][-1]))
    return params, r, nrmse


def SuperLearner_fun(X_valid, X_strain, y_valid, y_strain, r, nrmse, params, run_list, group_strain, group_valid, random_effects, classed=False, feature_selection=False):
    params_dict = {}

    xgbr = xgboost.XGBRegressor(seed=20, objective='reg:squarederror', booster='gbtree')
    for model in params:
        params_dict[model] = params[model][-1]
        params_dict[model] = {md: [params_dict[model][md]] for md in params_dict[model]}

    y_valid = y_valid.reshape(-1,)
    y_strain = y_strain.reshape(-1,)
    dict_models = {}
    if "xgb" in params:
        mod_xgb = GridSearchCV(estimator=xgbr, param_grid=params_dict["xgb"], scoring="r2", verbose=0,
                           n_jobs=-1, cv=5)
        dict_models["xgb"] = mod_xgb
    if "lasso" in params:
        mod_lasso = GridSearchCV(estimator=lasso_pipeline, param_grid=params_dict["lasso"], scoring="r2", verbose=0,
                           n_jobs=-1, cv=5)
        dict_models["lasso"] = mod_lasso
    if "rf" in params:
        mod_rf = GridSearchCV(estimator=rf, param_grid=params_dict["rf"], scoring="r2", verbose=0,
                           n_jobs=-1, cv=5)
        dict_models["rf"] = mod_rf
    if "svr" in params:
        mod_svr = GridSearchCV(estimator=svr_pipeline, param_grid=params_dict["svr"], scoring="r2", verbose=0,
                           n_jobs=-1, cv=5)
        dict_models["svr"] = mod_svr

    if "e_net" in params:
        mod_e_net = GridSearchCV(estimator=e_net_pipeline, param_grid=params_dict["e_net"], scoring="r2", verbose=0,
                           n_jobs=-1, cv=5)
        dict_models["e_net"] = mod_e_net

    if random_effects:
        gp_model = gpb.GPModel(group_data=group_strain, likelihood="gaussian")
    else:
        gp_model = gpb.GPModel(group_data=group_strain, group_rand_coef_data=X_strain[:, random_effects],
                    ind_effect_group_rand_coef=[1] * len(random_effects), likelihood="gaussian")
    data_train = gpb.Dataset(data=np.delete(X_strain, random_effects, axis=1), label=y_strain)

    yhat_strain, yhat_valid = [], []
    for model in run_list:
        if model != "super" and model != "gpb" and model != "merf":
            dict_models[model].fit(X_strain, y_strain)
            yhat = dict_models[model].predict(X_strain)
            yhat = yhat.reshape(-1,1)
            yhat_strain.append(yhat)
            yhat = dict_models[model].predict(X_valid)
            yhat = yhat.reshape(-1,1)
            yhat_valid.append(yhat)
        if model == "gpb":
            bst = gpb.train(params=params_dict['gpb'], train_set=data_train,
                            gp_model=gp_model, num_boost_round=200)
            if random_effects:
                pred = bst.predict(data=np.delete(X_strain, random_effects, axis=1), group_data_pred=group_strain,
                                   group_rand_coef_data_pred=X_strain[:, random_effects],
                               predict_var=True, pred_latent=False)
            else:
                pred = bst.predict(data=X_strain, group_data_pred=group_strain,
                               predict_var=True, pred_latent=False)
            yhat = pred["response_mean"].reshape(-1,1)
            yhat_strain.append(yhat)
            if random_effects:
                pred = bst.predict(data=np.delete(X_valid, random_effects, axis=1), group_data_pred=group_valid,
                                   group_rand_coef_data_pred=X_valid[:, random_effects],
                                    predict_var=True, pred_latent=False)
            else:
                pred = bst.predict(data=X_valid, group_data_pred=group_valid,
                               predict_var=True, pred_latent=False)
            yhat = pred["response_mean"].reshape(-1, 1)
            yhat_valid.append(yhat)
        if model =="merf":
            z = np.array([1] * len(X_strain)).reshape(-1, 1)
            z = np.hstack([z, X_strain[:, random_effects]])
            group_strain = group_strain.reset_index(drop=True)
            pred = merf.predict(X=X_strain, Z=z, clusters=group_strain)
            yhat = pred.reshape(-1,1)
            yhat_strain.append(yhat)

            z = np.array([1] * len(X_valid)).reshape(-1, 1)
            z = np.hstack([z, X_valid[:, random_effects]])
            group_valid = group_valid.reset_index(drop=True)
            pred = merf.predict(X=X_valid, Z=z, clusters=group_valid)
            yhat = pred.reshape(-1,1)
            yhat_valid.append(yhat)

    meta_X_strain = np.hstack(yhat_strain)
    meta_X_valid= np.hstack(yhat_valid)


    if not classed: mod_meta = GridSearchCV(estimator=svr_pipeline, param_grid=ensemble_params_grid, cv=5, scoring="r2", verbose=0, n_jobs=-1)
    else:
        gkf = list(GroupKFold(n_splits=5).split(meta_X_strain, y_strain, group_strain))
        mod_meta = GridSearchCV(estimator=svr_pipeline, param_grid=ensemble_params_grid, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

    results = mod_meta.fit(meta_X_strain, y_strain)
    ensemble_pred = mod_meta.predict(meta_X_valid)
    ensemble_pred = ensemble_pred.reshape(-1,)

    if "super" in params:
        params["super"].append(results.best_params_)
        r["super"].append(cor(y_valid, ensemble_pred))
        nrmse["super"].append(metrics.mean_squared_error(y_valid, ensemble_pred, squared=False) / statistics.stdev(y_valid))
    else:
        params["super"] = [results.best_params_]
        r["super"] = [cor(y_valid, ensemble_pred)]
        nrmse["super"] = [metrics.mean_squared_error(y_valid, ensemble_pred, squared=False) / statistics.stdev(y_valid)]
    print("Super_nrmse: " + str(nrmse["super"][-1]))
    print("Super_r: " + str(r["super"][-1]))
    print("Super_params: " + str(params["super"][-1]))
    return params, r, nrmse


def cv_with_arrays(df_ml, df_cv, val_splits, run_list, feature_selection=False, series_group = [], classed=False, random_effects = []):
    best_algorithms = []
    r_dict = {}
    nrmse_dict = {}
    all_params_dict = {}
    mean_r = {}
    z_r = {}
    mean_nrmse = {}
    df_r = pd.DataFrame()
    df_nrmse = pd.DataFrame()
    for col in range(df_cv.shape[1]):  # Jede spalte durchgehen
        print("Test fold: " + str(col))
        for inner_fold in range(val_splits):  # jede Zeile durchgehen
            df_valid = df_ml[df_cv.iloc[:, col] == inner_fold]  # X_valid erstellen
            df_strain = df_ml[(df_cv.iloc[:, col] != inner_fold) & (df_cv.iloc[:, col] != -1)]  # X strain ist ungleich x_valid und ungleich x_test
            if not series_group.empty:
                group_valid = series_group[df_cv.iloc[:, col] == inner_fold]  # group_valid erstellen
                group_strain = series_group[(df_cv.iloc[:, col] != inner_fold) & (
                            df_cv.iloc[:, col] != -1)]  # X strain ist ungleich x_valid und ungleich x_test
            X_valid = df_valid.iloc[:, :-1]
            X_strain = df_strain.iloc[:, :-1]
            y_valid = df_valid.iloc[:, [-1]]
            y_strain = df_strain.iloc[:, [-1]]

            if feature_selection == True:
                X_train_selected = fwiz.fit_transform(X_strain, y_strain)
                feature_selected = X_train_selected.columns.tolist()

                X_strain[X_strain.columns.difference(feature_selected)] = 0
                X_valid[X_valid.columns.difference(feature_selected)] = 0

            Z_loc = [X_strain.columns.get_loc(col) for col in random_effects]
            X_strain_selected, X_valid_selected, y_strain, y_valid = X_strain.values, X_valid.values, y_strain.values, y_valid.values
            print("Validation Fold: " + str(inner_fold))

            if "lasso" in run_list: all_params_dict, r_dict, nrmse_dict= find_params_lasso(X_valid=X_valid_selected, X_strain=X_strain_selected, y_valid=y_valid, y_strain=y_strain, params=all_params_dict, r=r_dict, nrmse=nrmse_dict, classed=classed, group_strain=group_strain)
            if "gpb" in run_list and not series_group.empty: all_params_dict, r_dict, nrmse_dict = find_params_gpb(X_valid=X_valid_selected, X_strain=X_strain_selected, y_valid=y_valid, y_strain=y_strain, group_valid=group_valid, group_strain=group_strain, random_effects=Z_loc, params=all_params_dict, r=r_dict, nrmse=nrmse_dict)
            if "merf" in run_list and not series_group.empty: r_dict, nrmse_dict = find_params_merf(X_valid=X_valid_selected, X_strain=X_strain_selected, y_valid=y_valid, y_strain=y_strain, group_valid=group_valid, group_strain=group_strain, random_effects=Z_loc, r=r_dict, nrmse=nrmse_dict)
            if "e_net" in run_list: all_params_dict, r_dict, nrmse_dict= find_params_e_net(X_valid=X_valid_selected, X_strain=X_strain_selected, y_valid=y_valid, y_strain=y_strain, params=all_params_dict, r=r_dict, nrmse=nrmse_dict, classed=classed, group_strain=group_strain)
            if "rf" in run_list: all_params_dict, r_dict, nrmse_dict = find_params_rf(X_valid=X_valid_selected, X_strain=X_strain_selected, y_valid=y_valid, y_strain=y_strain,  params=all_params_dict, r=r_dict, nrmse=nrmse_dict, classed=classed, group_strain=group_strain)
            if "svr" in run_list: all_params_dict, r_dict, nrmse_dict = find_params_svr(X_valid=X_valid_selected, X_strain=X_strain_selected, y_valid=y_valid, y_strain=y_strain, params=all_params_dict, r=r_dict, nrmse=nrmse_dict, classed=classed, group_strain=group_strain)
            if "xgb" in run_list: all_params_dict, r_dict, nrmse_dict = find_params_xgb(X_valid=X_valid_selected, X_strain=X_strain_selected, y_valid=y_valid, y_strain=y_strain, params=all_params_dict, r=r_dict, nrmse=nrmse_dict, classed=classed, group_strain=group_strain)
            if "super" in run_list: all_params_dict, r_dict, nrmse_dict = SuperLearner_fun(X_valid=X_valid_selected, X_strain=X_strain_selected, y_valid=y_valid, y_strain=y_strain, params=all_params_dict, r=r_dict, nrmse=nrmse_dict, run_list=run_list, group_strain=group_strain, group_valid=group_valid, random_effects=Z_loc, classed=classed, feature_selection=feature_selection)

        for model_name in run_list:  # Alle rs und nrmses sammeln in jeweils einem df
            mean_r[model_name] = np.mean(r_dict[model_name][-val_splits:])
            z_r[model_name] = np.mean(r_dict[model_name][-val_splits:]) / np.std(r_dict[model_name][-val_splits:])
            mean_nrmse[model_name] = np.mean(nrmse_dict[model_name][-val_splits:])
            print(model_name + " mean r: " + str(mean_r[model_name]))
            print(model_name + "mean nrmse: " + str(mean_nrmse[model_name]))
        print(mean_r)
        print(z_r)
        best_algorithms.append(sorted(mean_r, key=lambda key: mean_r[key])[-1])

    for model_name in run_list:                  # Alle rs und nrmses sammeln in jeweils einem df
        df_r[model_name] = r_dict[model_name]
        df_nrmse[model_name] = nrmse_dict[model_name]

    return df_r, df_nrmse, all_params_dict, best_algorithms


