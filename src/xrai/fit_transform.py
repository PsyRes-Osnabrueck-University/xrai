

class fit_transform:
        ############### From here on there is model output!!!!#################################################
    path = os.path.join(base_path, sub_folder_output)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.chdir(path)
    print(path)
    df_r.to_excel('r-inner-fold.xlsx')
    df_nrmse.to_excel('Nrmse-inner-fold.xlsx')

    # ------------------------------ Hyperparameter Tuning and final model (=EXTERNAL CROSS-VALIDATION)

    ################################################################
    last_feature = df_ml.columns.tolist()[-2]
    shaps = []
    r_list, nrmse_list, mae_list = [], [], []
    outcome_dict = {"true": [], "predicted": [],
                    "super_true": [], "super_predicted": []}
    dict_models = []
    id_list_super = []
    id_list = []
    X_list_super = []
    outcome = df_ml.columns.tolist()[-1]
    # best_algorithms = ["rf", "e_net", "merf", "merf", "e_net", "rf", "rf", "e_net", "rf", "svr"]

    for i, col in enumerate(df_nested_cv.columns.tolist()):  # Jede spalte durchgehen
        print("Test fold: " + str(col))
        df_y_train = df_ml.loc[df_nested_cv[col] != -1, [outcome]]
        df_y_test = df_ml.loc[df_nested_cv[col] == -1, [outcome]]

        df_X_train = df_ml.loc[df_nested_cv[col] != -1, :last_feature]
        df_X_test = df_ml.loc[df_nested_cv[col] == -1, :last_feature]

        df_fiX_train = df_X_train.drop(Z_list, axis=1)
        df_fiX_test = df_X_test.drop(Z_list, axis=1)

        df_Z_train = df_X_train[Z_list]
        df_Z_test = df_X_test[Z_list]

        group_train = df_id.loc[df_nested_cv[col] != -1, ["Class"]]
        group_test = df_id.loc[df_nested_cv[col] == -1, ["Class"]]

        session_train = df_id.loc[df_nested_cv[col] != -1, ["session"]]
        session_test = df_id.loc[df_nested_cv[col] == -1, ["session"]]

        Z_loc = [df_X_train.columns.get_loc(col) for col in Z_list]
        feature_list = df_X_train.columns.tolist()
        id_test = df_id.loc[df_nested_cv[col] == -1, :]
        id_test["fold"] = i
        id_test["learner"] = best_algorithms[i]
        dict_models.append({})

        # feature selection
        if feature_selection == True:
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

        if best_algorithms[i] == "svr" or best_algorithms[i] == "super" and "svr" in run_list:

            if not classed_splits:
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

        if best_algorithms[i] == "e_net" or best_algorithms[i] == "super" and "e_net" in run_list:
            ratios = np.arange(0.001, 0.3, 0.003)
            alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
                      1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 5, 10]

            if not classed_splits:
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

            if not classed_splits:
                mod_e_net = GridSearchCV(
                    estimator=e_net_pipeline, param_grid=e_net_pars, scoring="r2", cv=5, n_jobs=-1, verbose=0)
            else:
                mod_e_net = GridSearchCV(
                    estimator=e_net_pipeline, param_grid=e_net_pars, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

            dict_models[i]["e_net"] = mod_e_net

        if best_algorithms[i] == "lasso" or best_algorithms[i] == "super" and "lasso" in run_list:
            if not classed_splits:
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

            if not classed_splits:
                mod_lasso = GridSearchCV(
                    lasso_pipeline, param_grid=lasso_pars, cv=5, scoring="r2", verbose=0, n_jobs=-1)
            else:
                mod_lasso = GridSearchCV(
                    lasso_pipeline, param_grid=lasso_pars, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

            dict_models[i]["lasso"] = mod_lasso

        if best_algorithms[i] == "rf" or best_algorithms[i] == "super" and "rf" in run_list:

            if not classed_splits:
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
            if not classed_splits:
                mod_rf = GridSearchCV(
                    estimator=rf, param_grid=rf_pars, scoring="r2", cv=5, n_jobs=-1, verbose=0)
            else:
                mod_rf = GridSearchCV(
                    estimator=rf, param_grid=rf_pars, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

            dict_models[i]["rf"] = mod_rf

        if best_algorithms[i] == "xgb" or best_algorithms[i] == "super" and "xgb" in run_list:
            xgb_params_grid = {'max_depth': [2], 'learning_rate': [0.121], 'n_estimators': [
                100, 500], 'subsample': [0.5], 'colsample_bylevel': [0.275], 'colsample_bytree': [0.275]}
            max_depths = list(range(2, 10))
            xgb_params_grid["max_depth"] = max_depths

            if not classed_splits:
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
            if not classed_splits:
                super_xgb = GridSearchCV(
                    estimator=xgbr, param_grid=params_dict, scoring="r2", verbose=0, n_jobs=-1, cv=5)
            else:
                super_xgb = GridSearchCV(
                    estimator=xgbr, param_grid=params_dict, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

            results = super_xgb.fit(X_train_a, y_train_a, verbose=0)
            learning_rates = np.logspace(-3, -0.7, 3)
            params_dict = list_params(results.best_params_)
            params_dict["learning_rate"] = learning_rates

            if not classed_splits:
                super_xgb = GridSearchCV(
                    estimator=xgbr, param_grid=params_dict, scoring="r2", verbose=0, n_jobs=-1, cv=5)
            else:
                super_xgb = GridSearchCV(
                    estimator=xgbr, param_grid=params_dict, cv=gkf, scoring="r2", verbose=0, n_jobs=-1)

            results = super_xgb.fit(X_train_a, y_train_a, verbose=0)
            print(results.best_params_)
            xgb_pars = list_params(results.best_params_)
            if not classed_splits:
                mod_xgb = GridSearchCV(
                    estimator=xgbr, param_grid=xgb_pars, scoring="r2", verbose=0, n_jobs=-1, cv=5)
            else:
                mod_xgb = GridSearchCV(
                    estimator=xgbr, param_grid=xgb_pars, scoring="r2", verbose=0, n_jobs=-1, cv=gkf)
            dict_models[i]["xgb"] = mod_xgb

        if best_algorithms[i] == "gpb" or best_algorithms[i] == "super" and "gpb" in run_list:
            if Z_list:
                gp_model = gpb.GPModel(group_data=group_train, group_rand_coef_data=Z_train_a,
                                       ind_effect_group_rand_coef=[1]*len(Z_list), likelihood="gaussian")
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

        if best_algorithms[i] == "merf" or best_algorithms[i] == "super" and "merf" in run_list:
            z = np.array([1] * len(X_train_a)).reshape(-1, 1)
            z = np.hstack([z, Z_train_a])
            y_train_merf = y_train_a.reshape(-1, )
            group_train = group_train.reset_index(drop=True).squeeze()
            merf.fit(X=fiX_train_a, Z=z, clusters=group_train, y=y_train_merf)
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
                if not Z_list:
                    pred = dict_models[i][model].predict(data=fiX_train_a, group_data_pred=group_train,
                                                         predict_var=True, pred_latent=False)
                else:
                    pred = dict_models[i][model].predict(data=fiX_train_a, group_data_pred=group_train,
                                                         group_rand_coef_data_pred=Z_train_a,
                                                         predict_var=True, pred_latent=False)
                yhat = pred["response_mean"].reshape(-1, 1)
                yhat_train.append(yhat)
                if not Z_list:
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

        if best_algorithms[i] == "super":
            meta_X_train = np.hstack(yhat_train)
            meta_X_test = np.hstack(yhat_test)
            if not classed_splits:
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
        print("Auswahl: " + best_algorithms[i] + ": " + str(r_list[-1]))

        # hier wird der SHAP-Explainer aufgerufen, um die jeweiligen ML-Modelle zu erklären:
        if xAI == True:
            if best_algorithms[i] == "super":
                df_X_test.insert(loc=0, column="fold",
                                 value=[i]*len(df_X_test))
                df_X_test.insert(loc=1, column="Class",
                                 value=group_test.tolist())
                df_X_test.insert(loc=2, column="session",
                                 value=session_test["session"].tolist())
                # In der super_list sind alle ids, die über die Funktion ensemble_predict erklärt werden
                id_list_super.append(id_test)
                X_list_super.append(df_X_test)

            elif best_algorithms[i] == "rf" or best_algorithms[i] == "xgb":
                model_estimator = dict_models[i][best_algorithms[i]
                                                 ].best_estimator_
                explainer = shap.TreeExplainer(
                    model_estimator, data=shap.sample(X_train_a, 100))
            elif best_algorithms[i] == "gpb":
                explainer = shap.TreeExplainer(
                    bst, data=shap.sample(X_train_a, 100))
            elif best_algorithms[i] == "merf":
                explainer = shap.TreeExplainer(
                    merf.trained_fe_model, data=shap.sample(X_train_a, 100))
            else:
                explainer = shap.explainers.Permutation(
                    dict_models[i][best_algorithms[i]].predict, masker=shap.sample(X_train_a, 100), max_evals=600)
            if not best_algorithms[i] == "super":
                # In der id_list werden alle ids, die nicht vom superlearner vorhergesagt werden, gespeichert.
                id_list.append(id_test)
                if best_algorithms[i] == "rf" or best_algorithms[i] == "xgb" or best_algorithms[i] == "merf":
                    shaps.append(explainer(X_test_a, check_additivity=False))
                else:
                    shaps.append(explainer(X_test_a))

    if xAI == True:
        # Concatenate id and SHAP data from all prediction except super
        if any(model != "super" for model in best_algorithms):
            id_data = np.concatenate(id_list, axis=0)
            sh_values = shaps[0].values
            bs_values = shaps[0].base_values
            sh_data = shaps[0].data
            for i in range(1, len(shaps)):
                sh_values = np.concatenate(
                    (sh_values, np.array(shaps[i].values)), axis=0)
                bs_values = np.concatenate(
                    (bs_values, np.array(shaps[i].base_values)), axis=0)
                sh_data = np.concatenate(
                    (sh_data, np.array(shaps[i].data)), axis=0)
        # Calculation superlearner SHAP values
        if "super" in best_algorithms:
            id_super = pd.concat(id_list_super, ignore_index=True)
            X_test_super = pd.concat(X_list_super, ignore_index=True)
            explainer = shap.explainers.Permutation(
                ensemble_predict, masker=custom_masker, seed=1234, max_evals=600)
            shaps = []
            X_super = X_test_super.values
            shaps.append(explainer(X_super))
            # Concatenate superlearner shap values
            start = 0
            if sh_values is None:
                sh_values = shaps[0].values[:, 3:]
                bs_values = shaps[0].base_values
                sh_data = shaps[0].data[:, 3:]
                start = 1
            for i in range(start, len(shaps)):
                sh_values = np.concatenate(
                    (sh_values, np.array(shaps[i].values[:, 3:])), axis=0)
                bs_values = np.concatenate((bs_values, np.array(
                    shaps[i].base_values).reshape(-1,)), axis=0)
                sh_data = np.concatenate(
                    (sh_data, np.array(shaps[i].data[:, 3:])), axis=0)
            if id_data is None:
                id_data = id_super.values
            else:
                id_data = np.concatenate((id_data, np.array(id_super.values)))

        shap_values = shap.Explanation(values=sh_values,
                                       base_values=bs_values, data=sh_data,
                                       feature_names=feature_list)

    outcome_dict["true"] = np.concatenate(outcome_dict["true"], axis=0)
    outcome_dict["predicted"] = np.concatenate(
        outcome_dict["predicted"], axis=0)

    if "super" in best_algorithms:
        outcome_dict["super_true"] = np.concatenate(
            outcome_dict["super_true"], axis=0)
        outcome_dict["super_predicted"] = np.concatenate(
            outcome_dict["super_predicted"], axis=0)
        outcome_dict["true"] = np.concatenate(
            (outcome_dict["true"], outcome_dict["super_true"]), axis=0)
        outcome_dict["predicted"] = np.concatenate(
            (outcome_dict["predicted"], outcome_dict["super_predicted"]), axis=0)
    del outcome_dict["super_true"]
    del outcome_dict["super_predicted"]

    # Save rs and Nrmses
    df_results = pd.DataFrame(
        {"fold": [i for i in range(df_nested_cv.shape[1])],
            "n": [len(df_nested_cv.loc[df_nested_cv[col] == -1, "fold_0"]) for col in df_nested_cv.columns.tolist()],
         "r": r_list,
         "nrmse": nrmse_list,
         "mae": mae_list,
         "learner": best_algorithms})

    r_mean, r_lower, r_upper = mean_ci(
        df_results["r"], df_results["n"], limits="bootstrap")
    nrmse_mean, nrmse_lower, nrmse_upper = mean_ci(
        df_results["nrmse"], df_results["n"], limits="bootstrap")
    mae_mean, mae_lower, mae_upper = mean_ci(
        df_results["mae"], df_results["n"], limits="bootstrap")

    df_results_short = pd.DataFrame({"Value": ["mean", "95% lower", "95% upper"],
                                     "r": [r_mean, r_lower, r_upper],
                                     "nrmse": [nrmse_mean, nrmse_lower, nrmse_upper],
                                     "mae": [mae_mean, mae_lower, mae_upper]})

    writer = pd.ExcelWriter("Results.xlsx", engine="xlsxwriter")
    # Write each dataframe to a different worksheet.
    df_results.to_excel(writer, sheet_name="full results", index=False)
    df_results_short.to_excel(writer, sheet_name="short results", index=False)
    writer.close()

    # Save Beeswarm Plot
    if xAI == True:
        shap.summary_plot(shap_values, plot_size=(25, 15), max_display=15)

        shap.summary_plot(shap_values, plot_size=(
            25, 15), max_display=15, show=False)
        plt.savefig('summary_plot.png')
        plt.show()
        shap.plots.waterfall(shap_values[10], max_display=20, show=False)
        plt.gcf().set_size_inches(50, 15)
        plt.savefig('waterfall_plot.png')
        plt.show()
        shap.plots.heatmap(shap_values)

        # shap.plots.scatter(shap_values[:, "E3"], color=shap_values)
        # shap.plots.scatter(shap_values[:, "SU5"], color=shap_values)

    # Save SHAP values
    df_id_data = pd.DataFrame(
        id_data, columns=["Class", "session", "fold", "learner"])
    df_outcome = pd.DataFrame(outcome_dict)
    if xAI == True:
        df_sh_values = pd.DataFrame(sh_values, columns=feature_list)
        df_bs_values = pd.DataFrame(bs_values)
        df_sh_data = pd.DataFrame(sh_data, columns=feature_list)

    writer = pd.ExcelWriter("OUT.xlsx", engine="xlsxwriter")
    # Write each dataframe to a different worksheet.
    df_id_data.to_excel(writer, sheet_name="id", index=False)
    if xAI == True:
        df_sh_values.to_excel(writer, sheet_name="sh_values", index=False)
        df_bs_values.to_excel(writer, sheet_name="bs_values", index=False)
        df_sh_data.to_excel(writer, sheet_name="sh_data", index=False)
    df_outcome.to_excel(writer, sheet_name="Outcome", index=False)
    pd.DataFrame(feature_list, columns=["Feature_names"]).to_excel(
        writer, sheet_name="Features", index=False)
    writer.close()

    # SHAP IMPORTANCE values
    if xAI == True:
        global_shap_values = np.abs(shap_values.values).mean(0)
        df_shap_values = pd.DataFrame(
            global_shap_values.reshape(-1, len(global_shap_values)), columns=feature_list)

        df_shap_values_new = pd.DataFrame(
            {"Feature": feature_list, "SHAP-value": df_shap_values.iloc[0].tolist()})
        df_shap_values_new["percent_shap_value"] = df_shap_values_new["SHAP-value"] / df_shap_values_new[
            "SHAP-value"].sum() * 100
        dict_cors = {}
        n_rows = [len(df_id_data.loc[df_id_data["fold"] == fold, "fold"])
                  for fold in np.unique(df_id_data["fold"])]

        for feature in df_sh_values.columns.tolist():
            dict_cors[feature] = []
            for fold in np.unique(df_id_data["fold"]):
                shap_slice = df_sh_values.loc[df_id_data["fold"]
                                              == fold, feature]
                data_slice = df_sh_data.loc[df_id_data["fold"]
                                            == fold, feature]
                try:
                    dict_cors[feature].append(
                        cor(shap_slice.tolist(), data_slice.tolist()))
                except:
                    dict_cors[feature].append(0)

        df_cors = pd.DataFrame(dict_cors, index=range(len(best_algorithms)))
        dict_cors = {}
        for feature in df_cors.columns.tolist():
            dict_cors[feature] = []
            mean, lower_CI, higher_CI = mean_ci(
                df_cors[feature], np.array(n_rows), limits="bootstrap")
            dict_cors[feature].append(mean)
            dict_cors[feature].append(lower_CI)
            dict_cors[feature].append(higher_CI)
        df_cors_short = pd.DataFrame(dict_cors)
        df_shap_values_new["mean r"] = df_cors_short.loc[0].tolist()
        df_shap_values_new["lower 95%-CI"] = df_cors_short.loc[1].tolist()
        df_shap_values_new["higher 95%-CI"] = df_cors_short.loc[2].tolist()
        writer = pd.ExcelWriter("SHAP-IMPORTANCE.xlsx", engine="xlsxwriter")
        df_shap_values_new.to_excel(
            writer, sheet_name="SHAP importance", index=False)
        df_cors.to_excel(writer, sheet_name="Correlations", index=False)
        writer.close()
