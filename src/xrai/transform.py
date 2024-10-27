import scipy.stats
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
# import cv_fold
# import BerTopic.src.xrai.transform as transform

import gpboost as gpb

from featurewiz import FeatureWiz
from sklearn import metrics
import xgboost
import shap
import os
from merf import MERF

# import utils
from BerTopic.src.xrai.Preparation import Preparation

from BerTopic.src.xrai.utils import split_preparation, identify_last_column, list_params, mean_ci, cor
from BerTopic.src.xrai.cv_fold import cv_with_arrays

class Transform:
    def __init__(self,
                 shaps=[],
                 r_list=[],
                 nrmse_list=[],
                 mae_list=[],
                 outcome_dict={"true": [], "predicted": [],
                               "super_true": [], "super_predicted": []},
                 dict_models=[],
                 id_list_super=[],
                 id_list=[],
                 X_list_super=[],
                 dataPrepared=Preparation(base_path="/mnt/DATA/ACAD/KPP_HiWi/",
                                        #   base_path=os.getcwd(),
                                          file_name="distortions_final.xlsx",
                                          outcome=None,
                                          outcome_list=[],
                                          classed_splits=False,
                                          outcome_to_features=[],
                                          test_sets=10,
                                          val_sets=5)) -> None:

        xgbr = xgboost.XGBRegressor(
            seed=20, objective='reg:squarederror', booster='gbtree')

        lasso_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ('model', Lasso(random_state=42))
        ])

        e_net_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ('model', ElasticNet(random_state=42))
        ])

        ratios = np.arange(0.001, 0.3, 0.003)
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                  0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 5, 10]

        mod_e_net = GridSearchCV(e_net_pipeline,
                                 {"model__alpha": alphas,
                                  "model__l1_ratio": ratios},
                                 cv=5,
                                 scoring="r2",
                                 verbose=0,
                                 n_jobs=-1)
        rf = RandomForestRegressor(random_state=42)
        rf_params_grid = {
            'bootstrap': [True],
            'max_depth': [15],
            'max_features': [20],
            'min_samples_leaf': [2],
            'min_samples_split': [2, 4],
            'n_estimators': [1000]
        }

        merf = MERF(max_iterations=5)

        svr_pipeline = Pipeline([
            # Auch hier müssen wir zuerst skalieren
            ("scaler", StandardScaler()),
            ('model', SVR())
        ])

        svr_params_grid = {
            'model__kernel': ['rbf'],
            'model__C': [1, 2, 3],  # hatte 1 oder 2 als Optimum
            'model__degree': [2, 3],
            # hatte 0.001 als Optimum
            'model__coef0': [0.000001, 0.000005, 0.00001],
            'model__gamma': ['auto', 'scale']}

        mod_svr = GridSearchCV(
            estimator=svr_pipeline, param_grid=svr_params_grid, scoring="r2", cv=5, n_jobs=-1, verbose=0)

        # GPBOOST
        likelihood = "gaussian"

        gpb_param_grid = {'learning_rate': [0.1, 0.01],
                          'min_data_in_leaf': [10, 100],
                          'max_depth': [2, 5],
                          'lambda_l2': [0, 1, 10]}
        gpb_other_params = {'num_leaves': 2**10, 'verbose': 0}

        # xgboost
        xgb_params_grid = {'max_depth': [2, 3, 4],
                           'learning_rate': [0.001, 0.01, ],
                           'n_estimators': [5, 10, 20, 100, 500, 1000],
                           'subsample': [0.01, 0.1, 0.3],
                           'colsample_bylevel': [0.01, 0.1, 0.3],
                           # Muss evtl. weg!'early_stopping_rounds': [500]
                           'colsample_bytree': [0.01, 0.1, 0.3]}

        # Ensemble:
        ensemble_params_grid = {
            'model__kernel': ['rbf'],
            # hatte 1 oder 2 als Optimum
            'model__C': [1, 2, 3, 5, 6, 7, 8, 9, 10],
            'model__degree': [2, 3],
            # hatte 0.001 als Optimum
            'model__coef0': [1e-20, 1e-15, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'model__gamma': ['auto', 'scale']}

        mod_meta = GridSearchCV(estimator=svr_pipeline, param_grid=ensemble_params_grid, cv=5,
                                scoring="r2", verbose=0, n_jobs=-1)

        fwiz = FeatureWiz(corr_limit=0.8, feature_engg='', category_encoders='',
                          dask_xgboost_flag=False, nrows=None, verbose=0)

        dataPrepared.create_folds(Z_list=[],
                                  run_list=["merf", "super", "gpb"],
                                  val_sets=5,
                                  feature_selection=True,
                                  xAI=True)

        last_feature = dataPrepared.df_ml.columns.tolist()[-2]

        # Jede spalte durchgehen
        for i, col in enumerate(dataPrepared.df_nested_cv.columns.tolist()):
            print("Test fold: " + str(col))
            df_y_train = dataPrepared.df_ml.loc[dataPrepared.df_nested_cv[col] != -1, [
                dataPrepared.outcome]]
            df_y_test = dataPrepared.df_ml.loc[dataPrepared.df_nested_cv[col] == -1, [
                dataPrepared.outcome]]

            df_X_train = dataPrepared.df_ml.loc[dataPrepared.df_nested_cv[col]
                                                != -1, :last_feature]
            df_X_test = dataPrepared.df_ml.loc[dataPrepared.df_nested_cv[col]
                                               == -1, :last_feature]

            df_fiX_train = df_X_train.drop(dataPrepared.Z_list, axis=1)
            df_fiX_test = df_X_test.drop(dataPrepared.Z_list, axis=1)

            df_Z_train = df_X_train[dataPrepared.Z_list]
            df_Z_test = df_X_test[dataPrepared.Z_list]

            group_train = dataPrepared.df_id.loc[dataPrepared.df_nested_cv[col] != -1, [
                "Class"]]
            group_test = dataPrepared.df_id.loc[dataPrepared.df_nested_cv[col] == -1, [
                "Class"]]

            session_train = dataPrepared.df_id.loc[dataPrepared.df_nested_cv[col] != -1, [
                "session"]]
            session_test = dataPrepared.df_id.loc[dataPrepared.df_nested_cv[col] == -1, [
                "session"]]

            Z_loc = [df_X_train.columns.get_loc(
                col) for col in dataPrepared.Z_list]
            feature_list = df_X_train.columns.tolist()
            id_test = dataPrepared.df_id.loc[dataPrepared.df_nested_cv[col] == -1, :]
            id_test["fold"] = i
            id_test["learner"] = dataPrepared.best_algorithms[i]
            dict_models.append({})

            # feature selection
            if dataPrepared.feature_selection == True:
                X_train_selected = fwiz.fit_transform(df_X_train, df_y_train)
                feature_selected = X_train_selected.columns.tolist()

                df_X_train[df_X_train.columns.difference(feature_selected)] = 0
                df_X_test[df_X_test.columns.difference(feature_selected)] = 0

            y_train_a = df_y_train.values.reshape(-1,)
            y_test_a = df_y_test.values.reshape(-1,)

            X_test_a = df_X_test.values
            X_train_a = df_X_train.values

            fiX_train_a = df_fiX_train.values
            fiX_test_a = df_fiX_test.values

            Z_train_a = df_Z_train.values
            Z_test_a = df_Z_test.values

            if dataPrepared.best_algorithms[i] == "svr" or dataPrepared.best_algorithms[i] == "super" and "svr" in dataPrepared.run_list:

                if not dataPrepared.classed_splits:
                    super_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_params_grid, cv=5, scoring="r2", n_jobs=-1,
                                             verbose=0)
                else:
                    gkf = list(GroupKFold(n_splits=5).split(
                        X_train_a, y_train_a, group_train))
                    super_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_params_grid, cv=gkf, scoring="r2", verbose=0,
                                             n_jobs=-1)
                results = super_svr.fit(X_train_a, y_train_a)
                print(super_svr.best_params_)
                svr_pars = list_params(results.best_params_)
                mod_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_pars, scoring="r2",
                                       cv=5, n_jobs=-1, verbose=0)
                dict_models[i]["svr"] = mod_svr

            if dataPrepared.best_algorithms[i] == "e_net" or dataPrepared.best_algorithms[i] == "super" and "e_net" in dataPrepared.run_list:
                ratios = np.arange(0.001, 0.3, 0.003)
                alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
                          1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 5, 10]

                if not dataPrepared.classed_splits:
                    super_e_net = GridSearchCV(e_net_pipeline, {"model__alpha": alphas, "model__l1_ratio": ratios}, cv=5,
                                               scoring="r2", verbose=0, n_jobs=-1)
                else:
                    gkf = list(GroupKFold(n_splits=5).split(
                        X_train_a, y_train_a, group_train))
                    super_e_net = GridSearchCV(e_net_pipeline, {"model__alpha": alphas, "model__l1_ratio": ratios}, cv=gkf,
                                               scoring="r2", verbose=0, n_jobs=-1)
                results = super_e_net.fit(X_train_a, y_train_a)
                print(super_e_net.best_params_)
                e_net_pars = list_params(results.best_params_)

                if not dataPrepared.classed_splits:
                    mod_e_net = GridSearchCV(
                        estimator=e_net_pipeline, param_grid=e_net_pars, scoring="r2", cv=5, n_jobs=-1, verbose=0)
                else:
                    mod_e_net = GridSearchCV(
                        estimator=e_net_pipeline, param_grid=e_net_pars, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

                dict_models[i]["e_net"] = mod_e_net

            if dataPrepared.best_algorithms[i] == "lasso" or dataPrepared.best_algorithms[i] == "super" and "lasso" in dataPrepared.run_list:
                if not dataPrepared.classed_splits:
                    super_lasso = GridSearchCV(lasso_pipeline, {"model__alpha": np.arange(0.02, 2.0, 0.005)}, cv=5, scoring="r2",
                                               verbose=0, n_jobs=-1)
                else:
                    gkf = list(GroupKFold(n_splits=5).split(
                        X_train_a, y_train_a, group_train))
                    super_lasso = GridSearchCV(lasso_pipeline, {"model__alpha": np.arange(0.02, 2.0, 0.005)}, cv=gkf,
                                               scoring="r2", verbose=0, n_jobs=-1)
                results = super_lasso.fit(X_train_a, y_train_a)
                print(super_lasso.best_params_)
                lasso_pars = list_params(results.best_params_)

                if not dataPrepared.classed_splits:
                    mod_lasso = GridSearchCV(
                        lasso_pipeline, param_grid=lasso_pars, cv=5, scoring="r2", verbose=0, n_jobs=-1)
                else:
                    mod_lasso = GridSearchCV(
                        lasso_pipeline, param_grid=lasso_pars, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

                dict_models[i]["lasso"] = mod_lasso

            if dataPrepared.best_algorithms[i] == "rf" or dataPrepared.best_algorithms[i] == "super" and "rf" in dataPrepared.run_list:

                if not dataPrepared.classed_splits:
                    super_rf = GridSearchCV(
                        estimator=rf, param_grid=rf_params_grid, scoring="r2", cv=5, n_jobs=-1, verbose=0)
                else:
                    gkf = list(GroupKFold(n_splits=5).split(
                        X_train_a, y_train_a, group_train))
                    super_rf = GridSearchCV(
                        estimator=rf, param_grid=rf_params_grid, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

                results = super_rf.fit(X_train_a, y_train_a)
                print(super_rf.best_params_)
                rf_pars = list_params(results.best_params_)
                if not dataPrepared.classed_splits:
                    mod_rf = GridSearchCV(
                        estimator=rf, param_grid=rf_pars, scoring="r2", cv=5, n_jobs=-1, verbose=0)
                else:
                    mod_rf = GridSearchCV(
                        estimator=rf, param_grid=rf_pars, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

                dict_models[i]["rf"] = mod_rf

            if dataPrepared.best_algorithms[i] == "xgb" or dataPrepared.best_algorithms[i] == "super" and "xgb" in dataPrepared.run_list:
                xgb_params_grid = {'max_depth': [2], 'learning_rate': [0.121], 'n_estimators': [
                    100, 500], 'subsample': [0.5], 'colsample_bylevel': [0.275], 'colsample_bytree': [0.275]}
                max_depths = list(range(2, 10))
                xgb_params_grid["max_depth"] = max_depths

                if not dataPrepared.classed_splits:
                    super_xgb = GridSearchCV(
                        estimator=xgbr, param_grid=xgb_params_grid, scoring="r2", verbose=10, n_jobs=-1, cv=5)
                else:
                    gkf = list(GroupKFold(n_splits=5).split(
                        X_train_a, y_train_a, group_train))
                    super_xgb = GridSearchCV(
                        estimator=xgbr, param_grid=xgb_params_grid, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)
                results = super_xgb.fit(X_train_a, y_train_a, verbose=0)
                print(super_xgb.best_params_)

                subsamples = [0.1, 0.5, 1]
                colsample_bytrees = np.linspace(0.05, 0.5, 3)
                colsample_bylevel = np.linspace(0.05, 0.5, 3)

                # merge into full param dicts
                params_dict = xgb_params_grid
                params_dict["subsample"] = subsamples
                params_dict["colsample_bytree"] = colsample_bytrees
                params_dict["colsample_bylevel"] = colsample_bylevel
                if not dataPrepared.classed_splits:
                    super_xgb = GridSearchCV(
                        estimator=xgbr, param_grid=params_dict, scoring="r2", verbose=0, n_jobs=-1, cv=5)
                else:
                    super_xgb = GridSearchCV(
                        estimator=xgbr, param_grid=params_dict, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

                results = super_xgb.fit(X_train_a, y_train_a, verbose=0)
                learning_rates = np.logspace(-3, -0.7, 3)
                params_dict = list_params(results.best_params_)
                params_dict["learning_rate"] = learning_rates

                if not dataPrepared.classed_splits:
                    super_xgb = GridSearchCV(
                        estimator=xgbr, param_grid=params_dict, scoring="r2", verbose=0, n_jobs=-1, cv=5)
                else:
                    super_xgb = GridSearchCV(
                        estimator=xgbr, param_grid=params_dict, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

                results = super_xgb.fit(X_train_a, y_train_a, verbose=0)
                print(results.best_params_)
                xgb_pars = list_params(results.best_params_)
                if not dataPrepared.classed_splits:
                    mod_xgb = GridSearchCV(
                        estimator=xgbr, param_grid=xgb_pars, scoring="r2", verbose=0, n_jobs=-1, cv=5)
                else:
                    mod_xgb = GridSearchCV(
                        estimator=xgbr, param_grid=xgb_pars, scoring="r2", verbose=0, n_jobs=-1, cv=gkf)
                dict_models[i]["xgb"] = mod_xgb

            if dataPrepared.best_algorithms[i] == "gpb" or dataPrepared.best_algorithms[i] == "super" and "gpb" in dataPrepared.run_list:
                if dataPrepared.Z_list:
                    gp_model = gpb.GPModel(group_data=group_train, group_rand_coef_data=Z_train_a,
                                           ind_effect_group_rand_coef=[1]*len(dataPrepared.Z_list), likelihood="gaussian")
                else:
                    gp_model = gpb.GPModel(
                        group_data=group_train, likelihood="gaussian")

                data_train = gpb.Dataset(data=fiX_train_a, label=y_train_a)
                opt_params = gpb.grid_search_tune_parameters(param_grid=gpb_param_grid, params=gpb_other_params,
                                                             num_try_random=None, nfold=5, seed=1000, metric="rmse",
                                                             train_set=data_train, gp_model=gp_model,
                                                             use_gp_model_for_validation=True, verbose_eval=0,
                                                             num_boost_round=200, early_stopping_rounds=10)
                print(opt_params)
                bst = gpb.train(params=opt_params['best_params'], train_set=data_train,
                                gp_model=gp_model, num_boost_round=200)
                dict_models[i]["gpb"] = bst

            if dataPrepared.best_algorithms[i] == "merf" or dataPrepared.best_algorithms[i] == "super" and "merf" in dataPrepared.run_list:
                z = np.array([1] * len(X_train_a)).reshape(-1, 1)
                z = np.hstack([z, Z_train_a])
                y_train_merf = y_train_a.reshape(-1, )
                group_train = group_train.reset_index(drop=True).squeeze()
                merf.fit(X=fiX_train_a, Z=z,
                         clusters=group_train, y=y_train_merf)
                dict_models[i]["merf"] = merf

            yhat_train, yhat_test = [], []
            for model in dict_models[i]:
                if model != "gpb" and model != "merf" and model != "super":
                    dict_models[i][model].fit(X_train_a, y_train_a)
                    yhat = dict_models[i][model].predict(X_train_a)
                    yhat = yhat.reshape(-1, 1)
                    yhat_train.append(yhat)
                    yhat = dict_models[i][model].predict(X_test_a)
                    yhat = yhat.reshape(-1, 1)
                    yhat_test.append(yhat)
                if model == "gpb":
                    if not dataPrepared.Z_list:
                        pred = dict_models[i][model].predict(data=fiX_train_a, group_data_pred=group_train,
                                                             predict_var=True, pred_latent=False)
                    else:
                        pred = dict_models[i][model].predict(data=fiX_train_a, group_data_pred=group_train,
                                                             group_rand_coef_data_pred=Z_train_a,
                                                             predict_var=True, pred_latent=False)
                    yhat = pred["response_mean"].reshape(-1, 1)
                    yhat_train.append(yhat)
                    if not dataPrepared.Z_list:
                        pred = dict_models[i][model].predict(data=fiX_test_a, group_data_pred=group_test,
                                                             predict_var=True, pred_latent=False)
                    else:
                        pred = dict_models[i][model].predict(data=fiX_test_a, group_data_pred=group_test,
                                                             group_rand_coef_data_pred=Z_test_a,
                                                             predict_var=True, pred_latent=False)
                    yhat = pred["response_mean"].reshape(-1, 1)
                    yhat_test.append(yhat)
                if model == "merf":
                    z = np.array([1] * len(X_train_a)).reshape(-1, 1)
                    z = np.hstack([z, Z_train_a])
                    group_train = group_train.reset_index(drop=True).squeeze()
                    pred = dict_models[i][model].predict(
                        X=fiX_train_a, Z=z, clusters=group_train)
                    yhat = pred.reshape(-1, 1)
                    yhat_train.append(yhat)
                    z = np.array([1] * len(X_test_a)).reshape(-1, 1)
                    z = np.hstack([z, Z_test_a])
                    group_test = group_test.reset_index(drop=True).squeeze()
                    pred = dict_models[i][model].predict(
                        X=fiX_test_a, Z=z, clusters=group_test)
                    yhat = pred.reshape(-1, 1)
                    yhat_test.append(yhat)

            if dataPrepared.best_algorithms[i] == "super":
                meta_X_train = np.hstack(yhat_train)
                meta_X_test = np.hstack(yhat_test)
                if not dataPrepared.classed_splits:
                    mod_meta = GridSearchCV(
                        estimator=svr_pipeline, param_grid=ensemble_params_grid, cv=5, scoring="r2", verbose=0, n_jobs=-1)
                else:
                    gkf = list(GroupKFold(n_splits=5).split(
                        meta_X_train, y_train_a, group_train))
                    mod_meta = GridSearchCV(
                        estimator=svr_pipeline, param_grid=ensemble_params_grid, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)
                mod_meta.fit(meta_X_train, y_train_a)
                dict_models[i]["super"] = mod_meta
                pred = mod_meta.predict(meta_X_test)
                pred = pred.reshape(-1, )
                outcome_dict["super_true"].append(y_test_a)
                outcome_dict["super_predicted"].append(pred)
            else:
                pred = yhat_test[0]
                pred = pred.reshape(-1, )
                outcome_dict["true"].append(y_test_a)
                outcome_dict["predicted"].append(pred)
            r_list.append(cor(y_test_a, pred))
            nrmse_list.append(metrics.mean_squared_error(
                y_test_a, pred, squared=False) / statistics.stdev(y_test_a))
            mae_list.append(metrics.mean_absolute_error(y_test_a, pred))
            print("Auswahl: " +
                  dataPrepared.best_algorithms[i] + ": " + str(r_list[-1]))

            # hier wird der SHAP-Explainer aufgerufen, um die jeweiligen ML-Modelle zu erklären:
            if dataPrepared.xAI == True:
                if dataPrepared.best_algorithms[i] == "super":
                    df_X_test.insert(loc=0, column="fold",
                                     value=[i]*len(df_X_test))
                    df_X_test.insert(loc=1, column="Class",
                                     value=group_test.tolist())
                    df_X_test.insert(loc=2, column="session",
                                     value=session_test["session"].tolist())
                    # In der super_list sind alle ids, die über die Funktion ensemble_predict erklärt werden
                    id_list_super.append(id_test)
                    X_list_super.append(df_X_test)

                elif dataPrepared.best_algorithms[i] == "rf" or dataPrepared.best_algorithms[i] == "xgb":
                    model_estimator = dict_models[i][dataPrepared.best_algorithms[i]
                                                     ].best_estimator_
                    explainer = shap.TreeExplainer(
                        model_estimator, data=shap.sample(X_train_a, 100))
                elif dataPrepared.best_algorithms[i] == "gpb":
                    explainer = shap.TreeExplainer(
                        bst, data=shap.sample(X_train_a, 100))
                elif dataPrepared.best_algorithms[i] == "merf":
                    explainer = shap.TreeExplainer(
                        merf.trained_fe_model, data=shap.sample(X_train_a, 100))
                else:
                    explainer = shap.explainers.Permutation(
                        dict_models[i][dataPrepared.best_algorithms[i]].predict, masker=shap.sample(X_train_a, 100), max_evals=600)
                if not dataPrepared.best_algorithms[i] == "super":
                    # In der id_list werden alle ids, die nicht vom superlearner vorhergesagt werden, gespeichert.
                    id_list.append(id_test)
                    if dataPrepared.best_algorithms[i] == "rf" or dataPrepared.best_algorithms[i] == "xgb" or dataPrepared.best_algorithms[i] == "merf":
                        shaps.append(
                            explainer(X_test_a, check_additivity=False))
                    else:
                        shaps.append(explainer(X_test_a))

        if dataPrepared.xAI==True:
            # Concatenate id and SHAP data from all prediction except super
            if any(model != "super" for model in dataPrepared.best_algorithms):
                id_data = np.concatenate(id_list, axis=0)
                sh_values = shaps[0].values
                bs_values = shaps[0].base_values
                sh_data = shaps[0].data
                for i in range(1, len(shaps)):
                    sh_values = np.concatenate((sh_values, np.array(shaps[i].values)), axis=0)
                    bs_values = np.concatenate((bs_values, np.array(shaps[i].base_values)), axis=0)
                    sh_data = np.concatenate((sh_data, np.array(shaps[i].data)), axis=0)
            # Calculation superlearner SHAP values
            if "super" in dataPrepared.best_algorithms:
                id_super = pd.concat(id_list_super, ignore_index=True)
                X_test_super = pd.concat(X_list_super, ignore_index=True)
                explainer = shap.explainers.Permutation(ensemble_predict, masker=utils.custom_masker, seed=1234, max_evals=600) # max_evals should be double the size of prediction + error(3)
                shaps=[]
                X_super = X_test_super.values
                shaps.append(explainer(X_super))
                # Concatenate superlearner shap values
                start=0
                if sh_values is None:
                    sh_values = shaps[0].values[:, 3:]
                    bs_values = shaps[0].base_values
                    sh_data = shaps[0].data[:, 3:]
                    start=1
                for i in range(start, len(shaps)):
                    sh_values = np.concatenate((sh_values, np.array(shaps[i].values[:, 3:])), axis=0)
                    bs_values = np.concatenate((bs_values, np.array(shaps[i].base_values).reshape(-1,)), axis=0)
                    sh_data = np.concatenate((sh_data, np.array(shaps[i].data[:, 3:])), axis=0)
                if id_data is None: id_data = id_super.values
                else: id_data = np.concatenate((id_data, np.array(id_super.values)))

            shap_values = shap.Explanation(values=sh_values,
                                        base_values=bs_values, data=sh_data,
                                        feature_names=feature_list)

        outcome_dict["true"]=np.concatenate(outcome_dict["true"], axis=0)
        outcome_dict["predicted"]=np.concatenate(outcome_dict["predicted"], axis=0)

        if "super" in dataPrepared.best_algorithms:
            outcome_dict["super_true"]=np.concatenate(outcome_dict["super_true"], axis=0)
            outcome_dict["super_predicted"]=np.concatenate(outcome_dict["super_predicted"], axis=0)
            outcome_dict["true"]=np.concatenate((outcome_dict["true"], outcome_dict["super_true"]), axis=0)
            outcome_dict["predicted"]=np.concatenate((outcome_dict["predicted"], outcome_dict["super_predicted"]), axis=0)
        del outcome_dict["super_true"]
        del outcome_dict["super_predicted"]

        # Save rs and Nrmses
        df_results = pd.DataFrame(
                {"fold": [i for i in range(dataPrepared.df_nested_cv.shape[1])],
                "n": [len(dataPrepared.df_nested_cv.loc[dataPrepared.df_nested_cv[col]==-1, "fold_0"]) for col in dataPrepared.df_nested_cv.columns.tolist()],
                "r": r_list,
                "nrmse": nrmse_list,
                "mae": mae_list,
                "learner": dataPrepared.best_algorithms})

        r_mean, r_lower, r_upper = mean_ci(df_results["r"], df_results["n"], limits = "bootstrap")
        nrmse_mean, nrmse_lower, nrmse_upper = mean_ci(df_results["nrmse"], df_results["n"], limits = "bootstrap")
        mae_mean, mae_lower, mae_upper = mean_ci(df_results["mae"], df_results["n"], limits = "bootstrap")

        df_results_short = pd.DataFrame({"Value": ["mean", "95% lower", "95% upper"],
                                        "r": [r_mean, r_lower, r_upper],
                                        "nrmse": [nrmse_mean, nrmse_lower, nrmse_upper],
                                        "mae": [mae_mean, mae_lower, mae_upper]})
        writer = pd.ExcelWriter(os.path.join(dataPrepared.output_path, "Results.xlsx"), engine="xlsxwriter")
        # Write each dataframe to a different worksheet.
        df_results.to_excel(writer, sheet_name="full results", index=False)
        df_results_short.to_excel(writer, sheet_name="short results", index=False)
        writer.close()

        pass

    def ensemble_predict(self, X_test):
        fold = X_test[:, 0]
        change_id = np.where(fold[1:] != fold[:-1])[0]+1
        block_start_indices = [0] + change_id.tolist() + [len(fold)]
        block_arrays = []
        for i in range(len(block_start_indices) - 1):
            start = block_start_indices[i]
            end = block_start_indices[i + 1]
            block_values = X_test[start:end,:]
            block_arrays.append(block_values)
        y_list = []
        for idx, X in enumerate(block_arrays):
            yhat_list = []
            index = round(float(fold[block_start_indices[idx]]))
            group_test=X[:, 1]
            X=X[:, 3:]
            for model in dict_models[index]:
                if model != "gpb" and model != "merf" and model != "super":
                    yhat_list.append(dict_models[index][model].predict(X).reshape(-1,1))
                elif model == "gpb":
                    pred = dict_models[index][model].predict(data=np.delete(X, Z_loc, axis=1), group_data_pred=group_test,
                                                            group_rand_coef_data_pred=X[:, Z_loc],
                                    predict_var=True, pred_latent=False)
                    yhat = pred["response_mean"].reshape(-1, 1)
                    yhat_list.append(yhat)
                elif model == "merf":
                    z = np.array([1] * len(X)).reshape(-1, 1)
                    z = np.hstack([z, X[:, Z_loc]]).astype(float)
                    group_test = pd.Series(group_test.reshape(-1,))
                    yhat = dict_models[index][model].predict(X=np.delete(X, Z_loc, axis=1), Z=z, clusters=group_test)
                    yhat_list.append(yhat.reshape(-1, 1))
            meta_X_test = np.hstack(yhat_list)
            y_pred = dict_models[index]["super"].predict(meta_X_test)
            y_list.append(y_pred.reshape(-1,1))
        y = np.array(np.concatenate(y_list, axis=0))
        return y
