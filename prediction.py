import json
import scipy.stats
import statistics
import numpy as np
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning
import warnings
import cv_fold
import nexted_cv_pr
import gpboost as gpb


import sklearn.metrics
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from statistics import median

from featurewiz import FeatureWiz
from sklearn import metrics
import xgboost
import shap
import os
import sys

base_path = "C:/Users/clalk/JLUbox/Transkriptanalysen/2 TOPIC MODELING/Analysen/"
#base_path = r"C:\Users\clalk\JLUbox\Transkriptanalysen\3 N-GRAMME\Analysen"

sub_folder_Patient = "data/Patient_classed/hscl_aktuell_test"
#sub_folder_output = "data/Patient/hscl_diff" # oder "srs" oder "hscl_nächste_sitzung
sub_folder_output = "data/Patient_classed/hscl_aktuell_test"

scorer = "r2" # scorer = "neg_mean_squared_error
n_gramm_folder = r"C:\Users\clalk\JLUbox\Transkriptanalysen\3 N-GRAMME\Analysen"
n_gramm_folder_out = r"C:\Users\clalk\JLUbox\Transkriptanalysen\3 N-GRAMME\Analysen\out_lag_5"

Einsamkeit_folder = r"C:\Users\clalk\JLUbox\Transkriptanalysen\4 EINSAMKEIT"
Einsamkeit_folder_out = r"C:\Users\clalk\JLUbox\Transkriptanalysen\4 EINSAMKEIT\out"


def ensemble_predict(X_test):
    yhat_list = []
    Yyy.append(X_test)
    #print(X_test)
    for model in dict_models:
        if model != "gpb":
            yhat_list.append(dict_models[model].predict(X_test).reshape(-1,1))
        else:
            group_test_stacked = pd.concat([group_test]*2*X_test_a.shape[1], ignore_index=True)
            pred = bst.predict(data=X_test, group_data_pred=group_test,
                               predict_var=True, pred_latent=False)
            yhat = pred["response_mean"].reshape(-1, 1)
            yhat_list.append(yhat)
    meta_X_test = np.hstack(yhat_list)
    y_pred = mod_meta.predict(meta_X_test)
    y_pred = y_pred.reshape(-1,1)
    return y_pred



# Bei fertigem Datansatz AB HIER!!! --------------------------------------------------------------------------------------------------
outcome = "hscl_aktuelle_sitzung"  # Alternativ "srs_ges", hscl_aktuelle_sitzung, hscl_nächste_sitzung
path = os.path.join(base_path, sub_folder_Patient)
#path = n_gramm_folder
os.chdir(path)
print(path)

df_ml = pd.read_excel('data_hscl.xlsx')
df_nested_cv = pd.read_excel('hscl_cv.xlsx')
group = pd.read_excel('class.xlsx')
run_list = ["rf", "lasso", "xgb", "svr", "super", "e_net", "gpb"]  # additional "rf", "bart", "lasso", "cnn", "xgb", "super", "cnn"; super geht nur wenn alle anderen drin sind.
val_sets = 5
feature_selection = True
#df_ml["hscl_diff"] = df_ml["hscl_naechste_sitzung"]-df_ml["hscl_aktuelle_sitzung"]
#del df_ml["hscl_naechste_sitzung"]
#del df_ml["hscl_aktuelle_sitzung"]


df_r, df_nrmse, all_params, best_algorithms = cv_fold.cv_with_arrays(df_ml=df_ml, df_cv=df_nested_cv,
    val_splits=val_sets, run_list=run_list, feature_selection = feature_selection, series_group = group)

median(df_r["xgb"])
median(df_r["super"])
median(df_r["lasso"])
median(df_r["rf"])
median(df_r["svr"])
median(df_r["gpb"])

###############Ab hier gibt es Output, der gespeichert wird!!!!#################################################
path = os.path.join(base_path, n_gramm_folder_out)
os.chdir(path)
print(path)
df_r.to_excel('r-inner-fold.xlsx')
df_nrmse.to_excel('Nrmse-inner-fold.xlsx')

#------------------------------ Hyperparameter Tuning und finales Modell
def list_params(params_dict):
    params_dict = {md: [params_dict[md]] for md in params_dict}
    return params_dict

xgbr = xgboost.XGBRegressor(seed=20, objective='reg:squarederror', booster='gbtree')
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ('model', Lasso(random_state=42))
])

e_net_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ('model', ElasticNet(random_state=42))
])

ratios = np.arange(0.001, 0.3, 0.003)
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 5, 10]

mod_e_net = GridSearchCV(e_net_pipeline, {"model__alpha": alphas, "model__l1_ratio": ratios}, cv=5, scoring="r2", verbose=0, n_jobs=-1)
rf = RandomForestRegressor(random_state=42)
rf_params_grid = {
    'bootstrap': [True],
    'max_depth': [15],
    'max_features': [20],
    'min_samples_leaf': [2],
    'min_samples_split': [2, 4],
    'n_estimators': [1000]
}

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
              'min_data_in_leaf': [10,100,1000],
              'max_depth': [1,2,3,5,10],
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


def cor(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value
def custom_masker(mask, x):
    return (x * mask).reshape(1,len(x)) # in this simple example we just zero out the features we are masking


############ LÖSCHEN!!!!!!
run_list = ["lasso", "svr"]
feature_selection=True
last_feature = df_ml.columns.tolist()[-245]
################################################################

best_algorithms = ["super", "super"]
# last_feature = df_ml.columns.tolist()[-2]
shaps = []
r_list = []
nrmse_list = []
Yyy = []

for i, col in enumerate(df_nested_cv.columns.tolist()):  # Jede spalte durchgehen
    print("Test fold: " + str(col))
    df_y_train = df_ml.loc[df_nested_cv[col]!=-1, [outcome]]
    df_X_train = df_ml.loc[df_nested_cv[col]!=-1, :last_feature]
    group_train = group[df_nested_cv[col]!=-1]
    df_y_test = df_ml.loc[df_nested_cv[col]==-1, [outcome]]
    df_X_test = df_ml.loc[df_nested_cv[col]==-1, :last_feature]
    group_test = group[df_nested_cv[col] == -1]
    feature_list = df_X_train.columns.tolist()

    # feature selection
    if feature_selection == True:
        X_train_selected = fwiz.fit_transform(df_X_train, df_y_train)
        feature_selected = X_train_selected.columns.tolist()

        df_X_train[df_X_train.columns.difference(feature_selected)] = 0
        df_X_test[df_X_test.columns.difference(feature_selected)] = 0

    X_train_a = df_X_train.values
    y_train_a = df_y_train.values.reshape(-1,)
    X_test_a = df_X_test.values
    y_test_a = df_y_test.values.reshape(-1,)

    dict_models = {}

    if best_algorithms[i]=="svr" or best_algorithms[i]=="super" and "svr" in run_list:
        super_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_params_grid, scoring="r2", cv=5, n_jobs=-1, verbose=0)
        results = super_svr.fit(X_train_a, y_train_a)
        print(super_svr.best_params_)
        svr_pars = list_params(results.best_params_)
        mod_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_pars, scoring="r2",
                               cv=5, n_jobs=-1, verbose=0)
        dict_models["svr"]=mod_svr

    if best_algorithms[i]=="e_net" or best_algorithms[i]=="super" and "e_net" in run_list:
        ratios = np.arange(0.001, 0.3, 0.003)
        alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 5, 10]
        super_e_net = GridSearchCV(estimator=e_net_pipeline, param_grid={"model__alpha": alphas, "model__l1_ratio": ratios}, scoring="r2", cv=5, n_jobs=-1, verbose=0)
        results = super_e_net.fit(X_train_a, y_train_a)
        print(super_e_net.best_params_)
        e_net_pars = list_params(results.best_params_)
        mod_e_net = GridSearchCV(estimator=e_net_pipeline, param_grid=e_net_pars, scoring="r2",
                               cv=5, n_jobs=-1, verbose=0)
        dict_models["e_net"]=mod_e_net

    if best_algorithms[i]=="lasso" or best_algorithms[i]=="super" and "lasso" in run_list:
        super_lasso = GridSearchCV(lasso_pipeline, {"model__alpha": np.arange(0.001, 0.3, 0.005)}, cv=5, scoring="r2", verbose=0, n_jobs=-1)
        results = super_lasso.fit(X_train_a, y_train_a)
        print(super_lasso.best_params_)
        lasso_pars = list_params(results.best_params_)
        mod_lasso = GridSearchCV(lasso_pipeline, param_grid=lasso_pars, cv=5, scoring="r2",
                                 verbose=0, n_jobs=-1)
        dict_models["lasso"] = mod_lasso

    if best_algorithms[i]=="rf" or best_algorithms[i]=="super" and "rf" in run_list:
        super_rf = GridSearchCV(estimator=rf, param_grid=rf_params_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=0)
        results = super_rf.fit(X_train_a, y_train_a)
        print(super_rf.best_params_)
        rf_pars = list_params(results.best_params_)
        mod_rf = GridSearchCV(estimator=rf, param_grid=rf_pars, scoring="r2", cv=5, n_jobs=-1,
                              verbose=0)
        dict_models["rf"] = mod_rf

    if best_algorithms[i]=="xgb" or best_algorithms[i]=="super" and "xgb" in run_list:
        xgb_params_grid = {'max_depth': [2], 'learning_rate': [0.03], 'n_estimators': [1000], 'subsample': [0.4], 'colsample_bylevel': [0.1], 'colsample_bytree': [0.1]}
        max_depths = list(range(2, 4))
        xgb_params_grid["max_depth"] = max_depths
        super_xgb = GridSearchCV(estimator=xgbr, param_grid=xgb_params_grid, scoring="r2", verbose=0, n_jobs=-1, cv=2)
        results = super_xgb.fit(X_train_a, y_train_a, verbose=0)
        print(super_xgb.best_params_)

        subsamples = np.linspace(0.1, 1, 3)
        colsample_bytrees = np.linspace(0.1, 0.7, 3)
        colsample_bylevel = np.linspace(0.1, 0.7, 3)

        # merge into full param dicts
        params_dict = xgb_params_grid
        params_dict["subsample"] = subsamples
        params_dict["colsample_bytree"] = colsample_bytrees
        params_dict["colsample_bylevel"] = colsample_bylevel
        super_xgb = GridSearchCV(estimator=xgbr, param_grid=params_dict, scoring="r2", verbose=0, n_jobs=-1, cv=5)

        results = super_xgb.fit(X_train_a, y_train_a, verbose=0)
        learning_rates = np.logspace(-3, -0.7, 3)
        params_dict = list_params(results.best_params_)
        params_dict["learning_rate"] = learning_rates

        super_xgb = GridSearchCV(estimator=xgbr, param_grid=params_dict, scoring="r2", verbose=0, n_jobs=-1, cv=5)
        results = super_xgb.fit(X_train_a, y_train_a, verbose=0)
        print(results.best_params_)
        xgb_pars = list_params(results.best_params_)
        mod_xgb = GridSearchCV(estimator=xgbr, param_grid=xgb_pars, scoring="r2", verbose=0,
                               n_jobs=-1, cv=5)
        dict_models["xgb"] = mod_xgb

    if best_algorithms == "gpb" or best_algorithms[i]=="super" and "gpb" in run_list:
        gp_model = gpb.GPModel(group_data=group_train, likelihood="gaussian")
        data_train = gpb.Dataset(data=X_train_a, label=y_train_a)
        opt_params = gpb.grid_search_tune_parameters(param_grid=gpb_param_grid, params=gpb_other_params,
                                                     num_try_random=None, nfold=5, seed=1000, metric="rmse",
                                                     train_set=data_train, gp_model=gp_model,
                                                     use_gp_model_for_validation=True, verbose_eval=0,
                                                     num_boost_round=200, early_stopping_rounds=10)
        print(opt_params)
        dict_models["gpb"] = opt_params

    yhat_train, yhat_test = [], []
    for model in dict_models:
        if model != "gpb":
            dict_models[model].fit(X_train_a, y_train_a)
            yhat = dict_models[model].predict(X_train_a)
            yhat = yhat.reshape(-1,1)
            yhat_train.append(yhat)
            yhat = dict_models[model].predict(X_test_a)
            yhat = yhat.reshape(-1,1)
            yhat_test.append(yhat)
        else:
            bst = gpb.train(params=opt_params['best_params'], train_set=data_train,
                            gp_model=gp_model, num_boost_round=200)
            pred = bst.predict(data=X_train_a, group_data_pred=group_train,
                               predict_var=True, pred_latent=False)
            yhat = pred["response_mean"].reshape(-1,1)
            yhat_train.append(yhat)
            pred = bst.predict(data=X_test_a, group_data_pred=group_test,
                               predict_var=True, pred_latent=False)
            yhat = pred["response_mean"].reshape(-1, 1)
            yhat_test.append(yhat)

    if best_algorithms[i]=="super":

        meta_X_train = np.hstack(yhat_train)
        meta_X_test= np.hstack(yhat_test)
        mod_meta.fit(meta_X_train, y_train_a)
        pred = mod_meta.predict(meta_X_test)
        pred = pred.reshape(-1, )
    else:
        pred = yhat_test[0]
        pred = pred.reshape(-1, )

    r_list.append(cor(y_test_a, pred))
    nrmse_list.append(metrics.mean_squared_error(y_test_a, pred, squared=False) / statistics.stdev(y_test_a))
    print("Auswahl: " + best_algorithms[i] + ": " + str(metrics.mean_squared_error(y_test_a, pred, squared=False) / statistics.stdev(y_test_a)))

    # hier wird der SHAP-Explainer aufgerufen, um die jeweiligen ML-Modelle zu erklären:

    if best_algorithms[i]=="super": explainer = shap.explainers.Permutation(ensemble_predict, masker=custom_masker, seed=1234, max_evals=500)
    elif best_algorithms[i]=="gpb": explainer = shap.explainers.Permutation(bst, masker=shap.sample(X_train_a, 100), max_evals=503)
    else: explainer = shap.explainers.Permutation(dict_models[best_algorithms[i]].predict, masker=shap.sample(X_train_a, 100), max_evals=503)
    shaps.append(explainer(X_test_a))
    break

# Save rs and Nrmses
df_results = pd.DataFrame(
    {"r": r_list, "nrmse": nrmse_list, "learner": best_algorithms})
df_results.to_excel('Results.xlsx')




sh_values, bs_values, sh_data = [], [], []
sh_values = shaps[0].values
bs_values = shaps[0].base_values
sh_data = shaps[0].data
for i in range(len(shaps)):
    sh_values = np.concatenate((sh_values, np.array(shaps[i].values)), axis=0)
    bs_values = np.concatenate((bs_values, np.array(shaps[i].base_values)), axis=0)
    sh_data = np.concatenate((sh_data, np.array(shaps[i].data)), axis=0)

shap_values = shap.Explanation(values=sh_values,
                                   base_values=bs_values, data=sh_data,
                                   feature_names=feature_list)


# Save Beeswarm Plot
shap.summary_plot(shap_values, plot_size=(25, 15), max_display=8)

shap.summary_plot(shap_values, plot_size=(25, 15), max_display=8, show=False)
plt.savefig('summary_plot2.png')

shap.plots.waterfall(shap_values[2], max_display=5, show=False)
plt.gcf().set_size_inches(17,6)
plt.show()
plt.savefig('waterfall_plot.png')

shap.plots.scatter(shap_values[:,"personalizing"], show=False)
plt.gcf().set_size_inches(8,5)
plt.show()
plt.savefig('scatter_plot.png')

#Save SHAP values
df_sh_values = pd.DataFrame(sh_values)
df_sh_values.to_excel('sh_values.xlsx')
df_bs_values = pd.DataFrame(bs_values)
df_bs_values.to_excel('bs_values.xlsx')
df_sh_data = pd.DataFrame(sh_data)
df_sh_data.to_excel('sh_data.xlsx')

# SHAP IMPORTANCE values
global_shap_values = np.abs(shap_values.values).mean(0)
df_shap_values = pd.DataFrame(global_shap_values.reshape(-1, len(global_shap_values)), columns=feature_list)

df_shap_values_new = pd.DataFrame(
    {"Feature": feature_list, "SHAP-value": df_shap_values.iloc[0].tolist()})
df_shap_values_new["percent_shap_value"] = df_shap_values_new["SHAP-value"] / df_shap_values_new[
    "SHAP-value"].sum() * 100
df_shap_values_new.to_excel('SHAP-IMPORTANCE.xlsx')

# MUSS für jeden Datensatz nur einmal gemacht werden.
# Für CLASSSED DAtensätze: .../JLUbox/Transkriptanalysen/2 TOPIC MODELING/Analysen/creating_classed_dfs.R
#path = os.path.join(base_path, sub_folder_Patient)
path = Einsamkeit_folder
os.chdir(path)
print(path)

df = pd.read_excel('topic_modeling_outcome_patient.xlsx', index_col=0)
outcome = "hscl_naechste_sitzung"  # Welcher OUtcome soll vorhergesagt werden?
outcome_list = ["hscl_aktuelle_sitzung", "hscl_naechste_sitzung", "srs_ges"] # Welche Outcomes sind enthalten?
outcome_to_features = ["hscl_aktuelle_sitzung"] # Welche Outcomes sollen Features werden? Z.B. wenn ich hscl_aktuell
                        # als Prädiktor habe für hscl_lag_5
'''
Wir brauchen hier einen Datensatz, der als erstes die Variable Class enthält.
Diese beinhaltet den Patientencode. Danach kommt die Variable session. Diese 
beinhaltet die Sitzungsnummer. Danach kommen die Features. Diese können beliebig benannt sein. 
Anschließend kommen die Targets, also die abhängigen Variablen. Diese werden in die Liste
outcome_list eingetragen. 
'''

###### Session und Class anpassen
for line in range(len(df)):
    df.loc[line, "session"] = df.loc[line, "session"].partition("_")[2]

for line in range(len(df)):
    df.loc[line, "Class"] = df.loc[line, "Class"].partition("_")[0]

################################


test_sets, val_sets = 10, 5 # Anzahl an Test sets für outer cross-validation und Validation sets für inner cv
df_id, df_ml, df_nested_cv = nested_cv_preparation.split_preparation(test_splits=test_sets, val_splits=val_sets, df=df,
                                        outcome=outcome, outcome_list = outcome_list,
                                        outcome_to_features = outcome_to_features)  # Alternativ "srs_ges"

# File Benennung: Project (z.B. tm für topic modeling) _ Patient oder Therapeut oder Both _ classed oder nicht _ outcome (zB. hscl1 für hscl lag 1).
writer = pd.ExcelWriter("tm_patient_classed_hscl1.xlsx", engine="xlsxwriter")
# Write each dataframe to a different worksheet.
df_id.to_excel(writer, sheet_name="ID", index=False)
df_ml.to_excel(writer, sheet_name="ML", index=False)
df_nested_cv.to_excel(writer, sheet_name="CV", index=False)
writer.close()
# Close the Pandas Excel writer and output the Excel file.