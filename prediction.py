import scipy.stats
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from statistics import median

from sklearn import metrics
import xgboost
import shap
import os
shap.initjs()

base_path = "C:/Users/JLU-SU/JLUbox/Transkriptanalysen (Christopher Lalk)/2 TOPIC MODELING/Analysen/"
sub_folder_processing = "data/processing"
sub_folder_transkripte = "data/transkripte"
sub_folder_Patient = "data/Patient"
sub_folder_output = "data/Patient/hscl_nächste_sitzung" # oder "srs" oder "hscl_nächste_sitzung

def cor(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value

def scorer_fun(y_true, y_pred):
    return -metrics.mean_squared_error(y_true, y_pred, squared=True)

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value ** 2


def rsquared_dm(y_pred: np.ndarray, dtrain: xgboost.DMatrix) -> Tuple[str, float]:
    # y_pred[y_pred < -1] = -1 + 1e-6
    """ Return R^2 where x and y are array-like."""
    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_true)
    r_value = .03
    return r_value ** 2


def split_preparation(test_splits, val_splits, df, outcome, next=False):
    y = df[[outcome]]  # Outcome auswählen
    for column in df.columns:
        if "249" in column[0:3]: end_col = column  # letztes Topic = 249_... auswählen

    if next==True: hscl_akt = df["hscl_aktuelle_sitzung"]
    df = df.loc[:, :end_col]  # Response-Variablen entfernen
    df["hscl_aktuelle_sitzung"] = hscl_akt
    df[outcome] = y  # einzelne Response-Variable hinzufügen

    df = df.dropna()  # Missings fliegen raus!
    y = y.dropna()  # Auch bei y!

    test_kf = RepeatedKFold(n_splits=test_splits, n_repeats=1, random_state=42)
    val_kf = RepeatedKFold(n_splits=val_splits, n_repeats=1, random_state=42)

    for outer_fold in range(0,
                            test_splits):  # hinten dran kommt eine Variable für die folds. Darin steht in jedem Fold, wann man valid-set ist.
        df["fold_" + str(outer_fold)] = -1
    columns = df.columns.tolist()

    a_data = df.values

    for outer_fold, (train_index, test_index) in enumerate(test_kf.split(a_data)):  #
        a_train, a_test = a_data[train_index], a_data[test_index]
        print(outer_fold)
        inner_fold = 0
        for strain_index, valid_index in val_kf.split(a_train):
            print(inner_fold)
            a_strain, a_valid = a_train[strain_index], a_train[valid_index]
            df_valid = pd.DataFrame(a_valid, columns=columns)
            session_list = df_valid["session"].tolist()
            df.loc[df['session'].isin(session_list), "fold_" + str(
                outer_fold)] = inner_fold  # folds benennen, soweit eine row im valid-set ist
            inner_fold += 1
    df_cv = df.loc[:, "fold_0":]
    df_ml = df.loc[:, :outcome]
    # df_ml.insert(0, column, value)
    return df_ml, df_cv


def find_params_xgb(X_valid, X_strain, y_valid, y_strain, xgb_params, xgb_r2, xgb_nrmse):

    max_depths = list(range(2, 4))
    xgb_params_grid["max_depth"] = max_depths

    # pipeline für xgbr
    clf = GridSearchCV(estimator=xgbr,
                       param_grid=xgb_params_grid,
                       scoring='neg_mean_squared_error',
                       verbose=0, n_jobs=-1, cv=5)

    # training model
    results = clf.fit(X_strain, y_strain, verbose=0)


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
                       verbose=0, n_jobs=-1, cv=5)

    results = clf.fit(X_strain, y_strain, verbose=0)


    learning_rates = np.logspace(-3, -0.7, 3)
    params_dict = results.best_params_
    params_dict = {md: [params_dict[md]] for md in params_dict}
    params_dict["learning_rate"] = learning_rates

    clf = GridSearchCV(estimator=xgbr,
                       param_grid=params_dict,
                       scoring='neg_mean_squared_error',
                       verbose=0, n_jobs=-1, cv=5)

    results = clf.fit(X_strain, y_strain, verbose=0)
    print(results.best_params_)

    best_model = clf.best_estimator_
    xgb_params.append(results.best_params_)


    xgb_pred = best_model.predict(X_valid)
    xgb_pred = xgb_pred.reshape(-1,)
    y_valid = y_valid.reshape(-1, )
    xgb_r2.append(rsquared(y_valid, xgb_pred))
    xgb_nrmse.append(metrics.mean_squared_error(y_valid, xgb_pred, squared=False) / statistics.stdev(y_valid))
    print("XGB_nrmse: " + str(xgb_nrmse[-1]))
    return xgb_r2, xgb_params, xgb_nrmse


def find_params_rf(X_valid, X_strain, y_valid, y_strain, rf_params, rf_r2, rf_nrmse):
    mod_rf = GridSearchCV(estimator=rf,
                          param_grid=rf_params_grid,
                          cv=5,
                          n_jobs=-1,
                          verbose=0)
    y_strain = y_strain.reshape(-1, )
    results = mod_rf.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    rf_params.append(results.best_params_)

    rf_pred = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1, )
    rf_r2.append(rsquared(y_valid, rf_pred))
    rf_nrmse.append(metrics.mean_squared_error(y_valid, rf_pred, squared=False) / statistics.stdev(y_valid))
    print("rf_nrmse: " + str(rf_nrmse[-1]))
    return rf_r2, rf_params, rf_nrmse



def find_params_lasso(X_valid, X_strain, y_valid, y_strain, lasso_params, lasso_r2, lasso_nrmse, run="train"):
    results = mod_lasso.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    lasso_params.append(results.best_params_)
    lasso_pred = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1, )

    lasso_r2.append(rsquared(y_valid, lasso_pred))
    lasso_nrmse.append(metrics.mean_squared_error(y_valid, lasso_pred, squared=False) / statistics.stdev(y_valid))
    print("Lasso_nrmse: " + str(lasso_nrmse[-1]))
    return lasso_r2, lasso_params, lasso_nrmse



def find_params_svr(X_valid, X_strain, y_valid, y_strain, svr_params, svr_r2, svr_nrmse):

    mod_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_params_grid,
                           cv=3, n_jobs=-1, verbose=0)

    y_strain = y_strain.reshape(-1, )

    results = mod_svr.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    svr_params.append(results.best_params_)
    svr_pred = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1, )
    svr_pred = svr_pred.reshape(-1,)
    svr_r2.append(rsquared(y_valid, svr_pred))
    svr_nrmse.append(metrics.mean_squared_error(y_valid, svr_pred, squared=False) / statistics.stdev(y_valid))
    print("SVR_nrmse: " + str(svr_nrmse[-1]))
    return svr_r2, svr_params, svr_nrmse


def SuperLearner_fun(X_valid, X_strain, y_valid, y_strain, super_r2, super_nrmse, xgb_params, lasso_params,
                 rf_params,svr_params):
    params_dict = {}
    all_params = {"lasso": lasso_params, "xgb": xgb_params, "rf": rf_params, "svr": svr_params}
    for model in all_params:
        params_dict[model] = all_params[model][-1]
        params_dict[model] = {md: [params_dict[model][md]] for md in params_dict[model]}


    mod_xgb = GridSearchCV(estimator=xgbr, param_grid=params_dict["xgb"], scoring='neg_mean_squared_error', verbose=0, n_jobs=-1, cv=5)
    mod_lasso = GridSearchCV(estimator=lasso_pipeline, param_grid=params_dict["lasso"], cv=5, scoring="neg_mean_squared_error", verbose=0, n_jobs=-1)
    mod_rf = GridSearchCV(estimator=rf, param_grid=params_dict["rf"], scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=0)
    mod_svr = GridSearchCV(estimator=svr_pipeline, param_grid=params_dict["svr"], scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=0)

    y_valid = y_valid.reshape(-1,)
    y_strain = y_strain.reshape(-1,)

    dict_models = {"xgb": mod_xgb, "lasso": mod_lasso, "rf": mod_rf, "svr": mod_svr}

    yhat_strain, yhat_valid = [], []
    for model in dict_models:
        dict_models[model].fit(X_strain, y_strain)
        yhat = dict_models[model].predict(X_strain)
        yhat = yhat.reshape(-1,1)
        yhat_strain.append(yhat)
        yhat = dict_models[model].predict(X_valid)
        yhat = yhat.reshape(-1,1)
        yhat_valid.append(yhat)
    meta_X_strain = np.hstack(yhat_strain)
    meta_X_valid= np.hstack(yhat_valid)

    mod_meta.fit(meta_X_strain, y_strain)
    ensemble_pred = mod_meta.predict(meta_X_valid)
    ensemble_pred = ensemble_pred.reshape(-1,)

    super_r2.append(rsquared(y_valid, ensemble_pred))
    super_nrmse.append(metrics.mean_squared_error(y_valid, ensemble_pred, squared=False) / statistics.stdev(y_valid))
    print("superlearner_nrmse: " + str(super_nrmse[-1]))
    return super_r2, super_nrmse

def ensemble_predict(X_test):
    yhat_list = []
    for model in dict_models:
        yhat_list.append(dict_models[model].predict(X_test).reshape(-1,1))
    meta_X_test = np.hstack(yhat_list)
    y_pred = mod_meta.predict(meta_X_test)
    return y_pred

def cv_with_arrays(df_ml_to_array, df_cv_to_array, val_splits, run_list):
    best_algorithms = []
    array_ml, array_cv = df_ml_to_array.values, df_cv_to_array.values
    xgb_r2, xgb_params, xgb_nrmse = [], [], []
    lasso_r2, lasso_params, lasso_nrmse = [], [], []
    rf_r2, rf_params, rf_nrmse = [], [], []
    svr_r2, svr_params, svr_nrmse = [], [], []
    super_r2, super_nrmse= [], []
    out_r2 = pd.DataFrame()
    out_nrmse = pd.DataFrame()
    df_r2 = pd.DataFrame()
    df_nrmse = pd.DataFrame()
    for col in range(array_cv.shape[1]):  # Jede spalte durchgehen
        print("Test fold: " + str(col))
        for inner_fold in range(val_splits):  # jede Zeile durchgehen
            array_valid = array_ml[array_cv[:, col] == inner_fold]  # X_valid erstellen
            array_strain = array_ml[(array_cv[:, col] != inner_fold) & (array_cv[:, col] != -1)]  # X strain ist ungleich x_valid und ungleich x_test
            X_valid = array_valid[:, :-1]
            X_strain = array_strain[:, :-1]
            y_valid = array_valid[:, [-1]]
            y_strain = array_strain[:, [-1]]

            print("Validation Fold: " + str(inner_fold))

            if "lasso" in run_list: lasso_r2, lasso_params, lasso_nrmse= find_params_lasso(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain, lasso_params=lasso_params, lasso_r2=lasso_r2, lasso_nrmse=lasso_nrmse)
            if "rf" in run_list: rf_r2, rf_params, rf_nrmse = find_params_rf(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain, rf_params=rf_params, rf_r2=rf_r2, rf_nrmse=rf_nrmse)
            if "svr" in run_list: svr_r2, svr_params, svr_nrmse = find_params_svr(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain, svr_params=svr_params, svr_r2=svr_r2, svr_nrmse=svr_nrmse)
            if "xgb" in run_list: xgb_r2, xgb_params, xgb_nrmse = find_params_xgb(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain, xgb_params=xgb_params, xgb_r2=xgb_r2, xgb_nrmse=xgb_nrmse)
            if "super" in run_list: super_r2, super_nrmse = SuperLearner_fun(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain, super_r2=super_r2, super_nrmse=super_nrmse, xgb_params=xgb_params, lasso_params=lasso_params, rf_params=rf_params, svr_params=svr_params)

        r2s = {"lasso": lasso_r2[-val_splits:], "xgb": xgb_r2[-val_splits:], "rf": rf_r2[-val_splits:],
               "svr": svr_r2[-val_splits:], "super": super_r2[-val_splits:]}
        nrmses = {"lasso": lasso_nrmse[-val_splits:], "xgb": xgb_nrmse[-val_splits:], "rf": rf_nrmse[-val_splits:],
                  "svr": svr_nrmse[-val_splits:], "super": super_nrmse[-val_splits:]}
        mean_nrmse = {"lasso": np.mean(lasso_nrmse[-val_splits:]), "xgb": np.mean(xgb_nrmse[-val_splits:]), "rf": np.mean(rf_nrmse[-val_splits:]),
               "svr": np.mean(svr_nrmse[-val_splits:]), "super": np.mean(super_nrmse[-val_splits:])}
        print(mean_nrmse)
        best_algorithms.append(sorted(mean_nrmse, key=lambda key: mean_nrmse[key])[0])

        for model_name in run_list:  # Alle R2s und nrmses sammeln in jeweils einem df
            out_r2[model_name] = r2s[model_name]
            out_nrmse[model_name] = nrmses[model_name]
            print(model_name + " median r2: " + str(median(out_r2[model_name])))
            print(model_name + " median nrmse: " + str(median(out_nrmse[model_name])))

    r2s = {"lasso": lasso_r2, "xgb": xgb_r2, "rf": rf_r2, "svr": svr_r2, "super": super_r2}
    nrmses = {"lasso": lasso_nrmse, "xgb": xgb_nrmse, "rf": rf_nrmse, "svr": svr_nrmse, "super": super_nrmse}
    all_params = {"lasso": lasso_params, "xgb": xgb_params, "rf": rf_params, "svr": svr_params}

    for model_name in run_list:                  # Alle R2s und nrmses sammeln in jeweils einem df
        df_r2[model_name] = r2s[model_name]
        df_nrmse[model_name] = nrmses[model_name]

    return df_r2, df_nrmse, all_params, lasso_params, xgb_params, rf_params, svr_params, best_algorithms


# xgboost model
xgb_params_grid = {'max_depth': [2],
                   'learning_rate': [0.03],
                   'n_estimators': [1000],
                   'subsample': [0.4],
                   'colsample_bylevel': [0.1],
                   'colsample_bytree': [0.1]}  # Muss evtl. weg!'early_stopping_rounds': [500]

xgbr = xgboost.XGBRegressor(seed=20, objective='reg:squarederror', booster='gbtree')

mod_xgb = GridSearchCV(estimator=xgbr, param_grid=xgb_params_grid, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1, cv=5)


# Lasso Model
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ('model', Lasso(random_state=42))
])
nur_lasso = Lasso()

mod_lasso = GridSearchCV(lasso_pipeline, {"model__alpha": np.arange(0.02, 0.7, 0.005)}, cv=5, scoring="neg_mean_squared_error", verbose=0, n_jobs=-1)

# Random Forest Model
rf_params_grid = {
    'bootstrap': [True],
    'max_depth': [15, 20],
    'max_features': [20, 50],
    'min_samples_leaf': [2, 3],
    'min_samples_split': [2, 4],
    'n_estimators': [1000]
}

rf = RandomForestRegressor(random_state=42)

mod_rf = GridSearchCV(estimator=rf, param_grid=rf_params_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=0)

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

mod_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_params_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=0)

# Ensemble:

mod_meta = GridSearchCV(estimator=lasso_pipeline, param_grid={"model__alpha": np.arange(0.001, 0.5, 0.005)}, cv=10,
                            scoring="neg_mean_squared_error", verbose=0, n_jobs=-1)


# Bei fertigem Datansatz AB HIER!!! --------------------------------------------------------------------------------------------------
outcome = "hscl_naechste_sitzung"  # Alternativ "srs_ges"
path = os.path.join(base_path, sub_folder_Patient)
os.chdir(path)
print(path)
df_ml = pd.read_excel('data_hscl_next.xlsx')
df_nested_cv = pd.read_excel('CV_folds_hscl_next.xlsx')
run_list = ["rf", "lasso", "xgb", "svr", "super"]  # additional "rf", "bart", "lasso", "cnn", "xgb", "super", "cnn"; super geht nur wenn alle anderen drin sind.
val_sets = 5

df_r2, df_nrmse, all_params, lasso_params, xgb_params, rf_params, svr_params, best_algorithms = cv_with_arrays(
    df_ml_to_array=df_ml, df_cv_to_array=df_nested_cv,
    val_splits=val_sets, run_list=run_list)


median(df_r2["xgb"])
median(df_r2["super"])
median(df_r2["lasso"])
median(df_r2["rf"])
median(df_r2["svr"])

###############Ab hier gibt es Output, der gespeichert wird!!!!#################################################
path = os.path.join(base_path, sub_folder_output)
os.chdir(path)
print(path)
df_r2.to_excel('R2-inner-fold.xlsx')
df_nrmse.to_excel('Nrmse-inner-fold.xlsx')


#------------------------------ Hyperparameter Tuning und finales Modell
def list_params(params_dict):
    params_dict = {md: [params_dict[md]] for md in params_dict}
    return params_dict

def calculate_fit():
    y_pred = results.best_estimator_.predict(X_test_a)
    print(metrics.mean_squared_error(y_test_a, y_pred, squared=False) / statistics.stdev(y_test_a))
    print(rsquared(y_pred, y_test_a))



last_feature = df_ml.columns.tolist()[-2]
shaps = []
r2_list = []
nrmse_list = []


for i, col in enumerate(df_nested_cv.columns.tolist()):  # Jede spalte durchgehen
    print("Test fold: " + str(col))
    df_y_train = df_ml.loc[df_nested_cv[col]!=-1, [outcome]]
    df_X_train = df_ml.loc[df_nested_cv[col]!=-1, :last_feature]
    df_y_test = df_ml.loc[df_nested_cv[col]==-1, [outcome]]
    df_X_test = df_ml.loc[df_nested_cv[col]==-1, :last_feature]
    X_train_a = df_X_train.values
    y_train_a = df_y_train.values.reshape(-1,)
    X_test_a = df_X_test.values
    y_test_a = df_y_test.values.reshape(-1,)
    feature_list = df_X_train.columns.tolist()

    dict_models = {}

    if best_algorithms[i]=="svr" or best_algorithms[i]=="super":
        super_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_params_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=0)
        results = super_svr.fit(X_train_a, y_train_a)
        print(super_svr.best_params_)
        svr_pars = list_params(results.best_params_)
        mod_svr = GridSearchCV(estimator=svr_pipeline, param_grid=svr_pars, scoring="neg_mean_squared_error",
                               cv=5, n_jobs=-1, verbose=0)
        dict_models["svr"]=mod_svr

    if best_algorithms[i]=="lasso" or best_algorithms[i]=="super":
        super_lasso = GridSearchCV(lasso_pipeline, {"model__alpha": np.arange(0.001, 0.3, 0.005)}, cv=5, scoring="neg_mean_squared_error", verbose=0, n_jobs=-1)
        results = super_lasso.fit(X_train_a, y_train_a)
        print(super_lasso.best_params_)
        lasso_pars = list_params(results.best_params_)
        mod_lasso = GridSearchCV(lasso_pipeline, param_grid=lasso_pars, cv=5, scoring="neg_mean_squared_error",
                                 verbose=0, n_jobs=-1)
        dict_models["lasso"] = mod_lasso

    if best_algorithms[i]=="rf" or best_algorithms[i]=="super":
        super_rf = GridSearchCV(estimator=rf, param_grid=rf_params_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=0)
        results = super_rf.fit(X_train_a, y_train_a)
        print(super_rf.best_params_)
        rf_pars = list_params(results.best_params_)
        mod_rf = GridSearchCV(estimator=rf, param_grid=rf_pars, scoring="neg_mean_squared_error", cv=5, n_jobs=-1,
                              verbose=0)
        dict_models["rf"] = mod_rf

    if best_algorithms[i]=="xgb" or best_algorithms[i]=="super":
        xgb_params_grid = {'max_depth': [2], 'learning_rate': [0.03], 'n_estimators': [1000], 'subsample': [0.4], 'colsample_bylevel': [0.1], 'colsample_bytree': [0.1]}
        max_depths = list(range(2, 4))
        xgb_params_grid["max_depth"] = max_depths
        super_xgb = GridSearchCV(estimator=xgbr, param_grid=xgb_params_grid, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1, cv=2)
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
        super_xgb = GridSearchCV(estimator=xgbr, param_grid=params_dict, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1, cv=5)

        results = super_xgb.fit(X_train_a, y_train_a, verbose=0)
        print(super_xgb.best_params_)
        learning_rates = np.logspace(-3, -0.7, 3)
        params_dict = list_params(results.best_params_)
        params_dict["learning_rate"] = learning_rates

        super_xgb = GridSearchCV(estimator=xgbr, param_grid=params_dict, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1, cv=5)
        results = super_xgb.fit(X_train_a, y_train_a, verbose=0)
        print(results.best_params_)
        xgb_pars = list_params(results.best_params_)
        mod_xgb = GridSearchCV(estimator=xgbr, param_grid=xgb_pars, scoring='neg_mean_squared_error', verbose=0,
                               n_jobs=-1, cv=5)
        dict_models["xgb"] = mod_xgb

    yhat_train, yhat_test = [], []
    for model in dict_models:
        dict_models[model].fit(X_train_a, y_train_a)
        yhat = dict_models[model].predict(X_train_a)
        yhat = yhat.reshape(-1,1)
        yhat_train.append(yhat)
        yhat = dict_models[model].predict(X_test_a)
        yhat = yhat.reshape(-1,1)
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

    r2_list.append(rsquared(y_test_a, pred))
    nrmse_list.append(metrics.mean_squared_error(y_test_a, pred, squared=False) / statistics.stdev(y_test_a))
    print("Auswahl: " + best_algorithms[i] + ": " + str(metrics.mean_squared_error(y_test_a, pred, squared=False) / statistics.stdev(y_test_a)))

    if best_algorithms[i]=="super": explainer = shap.explainers.Permutation(ensemble_predict, masker=shap.sample(X_train_a, 100), max_evals=503)
    else: explainer = shap.explainers.Permutation(dict_models[best_algorithms[i]].predict, masker=shap.sample(X_train_a, 100), max_evals=503)
    shaps.append(explainer(X_test_a))

# Save R2s and Nrmses
df_results = pd.DataFrame(
    {"r2": r2_list, "nrmse": nrmse_list, "learner": best_algorithms})
df_results.to_excel('Results.xlsx')

sh_values, bs_values, sh_data = None, None, None
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
shap.summary_plot(shap_values, plot_size=(25, 22), max_display=20, show=False)
plt.savefig('summary_plot.png')

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


# MUSS für jeden Datensatz nur einmal gemacht werden -------------------------------------------------------------------------
path = os.path.join(base_path, sub_folder_data)
os.chdir(path)
print(path)
df = pd.read_excel('patient_diagnose_5_250_patientenebene_zufaellig.xlsx', index_col=0)

test_sets, val_sets = 10, 5
outcome = "hscl_naechste_sitzung"  # Alternativ "srs_ges"

df_ml, df_nested_cv = split_preparation(test_splits=test_sets, val_splits=val_sets, df=df,
                                        outcome=outcome, next=True)  # Alternativ "srs_ges"
df_ml = df_ml.iloc[:, 1:]  # erste Spalte löschen (session-Variable ist nicht ml-geeignet)

df_ml.to_excel("data_hscl_next.xlsx", index=False)
df_nested_cv.to_excel("CV_folds_hscl_next.xlsx", index=False)

