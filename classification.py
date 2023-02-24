import scipy.stats
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

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
sub_folder_output = "data/Patient/diagnose" # oder "srs" oder "hscl_nächste_sitzung
sub_folder_data = "data"


def split_preparation(test_splits, val_splits, df, outcome):
    y = df[[outcome]]  # Outcome auswählen
    for column in df.columns:
        if "249" in column[0:3]: end_col = column  # letztes Topic = 249_... auswählen

    df = df.loc[:, :end_col]  # Response-Variablen entfernen
    df[outcome] = y  # einzelne Response-Variable hinzufügen

    df = df.dropna()  # Missings fliegen raus!
    y = y.dropna()  # Auch bei y!

    test_kf = RepeatedStratifiedKFold(n_splits=test_splits, n_repeats=1, random_state=42)
    val_kf = RepeatedStratifiedKFold(n_splits=val_splits, n_repeats=1, random_state=42)

    for outer_fold in range(0, test_splits):  # hinten dran kommt eine Variable für die folds. Darin steht in jedem Fold, wann man valid-set ist.
        df["fold_" + str(outer_fold)] = -1
    df_zw = df.loc[:, :outcome]
    df_X = df_zw.iloc[:, :-1]
    df_y = df_zw.iloc[:, [-1]]
    columns = df_X.columns.tolist()

    a_X = df_X.values
    a_y = df_y.values

    for outer_fold, (train_index, test_index) in enumerate(test_kf.split(a_X, a_y)):  #
        a_X_train, a_X_test = a_X[train_index], a_X[test_index]
        a_y_train, a_y_test = a_y[train_index], a_y[test_index]
        print(outer_fold)
        inner_fold = 0
        for strain_index, valid_index in val_kf.split(a_X_train, a_y_train):
            print(inner_fold)
            a_X_strain, a_X_valid = a_X_train[strain_index], a_X_train[valid_index]
            a_y_strain, a_y_valid = a_y_train[strain_index], a_y_train[valid_index]

            df_valid = pd.DataFrame(a_X_valid, columns=columns)
            session_list = df_valid["session"].tolist()
            df.loc[df['session'].isin(session_list), "fold_" + str(
                outer_fold)] = inner_fold  # folds benennen, soweit eine row im valid-set ist
            inner_fold += 1
    df_ml = df.loc[:, :outcome]
    df_cv = df.loc[:, "fold_0":]

    # df_ml.insert(0, column, value)
    return df_ml, df_cv


def find_params_xgb(X_valid, X_strain, y_valid, y_strain, xgb_params, xgb_f1):

    max_depths = list(range(2, 4))
    xgb_params_grid["max_depth"] = max_depths
    y_strain = y_strain.reshape(-1,1)
    y_strain_trans = label_encoder.transform(y_strain)

    # pipeline für xgbr
    clf = GridSearchCV(estimator=xgbc,
                       param_grid=xgb_params_grid,
                       scoring='f1_weighted',
                       verbose=0, n_jobs=-1, cv=5)

    # training model
    results = clf.fit(X_strain, y_strain_trans, verbose=0)


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

    clf = GridSearchCV(estimator=xgbc,
                       param_grid=params_dict,
                       scoring='f1_weighted',
                       verbose=0, n_jobs=-1, cv=5)

    results = clf.fit(X_strain, y_strain_trans, verbose=0)


    learning_rates = np.logspace(-3, -0.7, 3)
    params_dict = results.best_params_
    params_dict = {md: [params_dict[md]] for md in params_dict}
    params_dict["learning_rate"] = learning_rates

    clf = GridSearchCV(estimator=xgbc,
                       param_grid=params_dict,
                       scoring='f1_weighted',
                       verbose=0, n_jobs=-1, cv=5)

    results = clf.fit(X_strain, y_strain_trans, verbose=0)
    print(results.best_params_)

    best_model = clf.best_estimator_
    xgb_params.append(results.best_params_)


    xgb_pred = best_model.predict(X_valid)
    xgb_pred = label_encoder.inverse_transform(xgb_pred)
    xgb_pred = xgb_pred.reshape(-1,1)
    y_valid = y_valid.reshape(-1, 1)
    xgb_f1.append(f1_score(y_valid, xgb_pred, average="weighted"))
    print("XGB_f1: " + str(xgb_f1[-1]))
    return xgb_f1, xgb_params


def find_params_rf(X_valid, X_strain, y_valid, y_strain, rf_params, rf_f1):
    mod_rf = GridSearchCV(estimator=rf, param_grid=rf_params_grid, scoring="f1_weighted", cv=5, n_jobs=-1, verbose=0)
    y_strain = y_strain.reshape(-1, )
    results = mod_rf.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    rf_params.append(results.best_params_)

    rf_pred = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1, )
    rf_f1.append(f1_score(y_valid, rf_pred, average="weighted"))
    print("rf_f1: " + str(rf_f1[-1]))
    return rf_f1, rf_params

def find_params_svc(X_valid, X_strain, y_valid, y_strain, svc_params, svc_f1):

    mod_svc = GridSearchCV(estimator=svc_pipeline, param_grid=svc_params_grid, scoring="f1_weighted",
                           cv=3, n_jobs=-1, verbose=0)

    y_strain = y_strain.reshape(-1, )

    results = mod_svc.fit(X_strain, y_strain)
    best_model = results.best_estimator_
    svc_params.append(results.best_params_)
    svc_pred = best_model.predict(X_valid)
    y_valid = y_valid.reshape(-1, )
    svc_pred = svc_pred.reshape(-1,)
    svc_f1.append(f1_score(y_valid, svc_pred, average="weighted"))
    print("SVC_f1: " + str(svc_f1[-1]))
    return svc_f1, svc_params


def SuperLearner_fun(X_valid, X_strain, y_valid, y_strain, super_f1, xgb_params,
                 rf_params,svc_params):
    params_dict = {}
    all_params = {"xgb": xgb_params, "rf": rf_params, "svc": svc_params}
    for model in all_params:
        params_dict[model] = all_params[model][-1]
        params_dict[model] = {md: [params_dict[model][md]] for md in params_dict[model]}


    mod_xgb = GridSearchCV(estimator=xgbc, param_grid=params_dict["xgb"], scoring='f1_weighted', verbose=0, n_jobs=-1, cv=5)
    mod_rf = GridSearchCV(estimator=rf, param_grid=params_dict["rf"], scoring="f1_weighted", cv=5, n_jobs=-1, verbose=0)
    mod_svc = GridSearchCV(estimator=svc_pipeline, param_grid=params_dict["svc"], scoring="f1_weighted", cv=5, n_jobs=-1, verbose=0)

    y_valid = y_valid.reshape(-1,)
    y_strain = y_strain.reshape(-1,)

    dict_models = {"xgb": mod_xgb, "rf": mod_rf, "svc": mod_svc}

    yhat_strain, yhat_valid = [], []
    for model in dict_models:
        if model != "xgb":
            dict_models[model].fit(X_strain, y_strain)
            yhat = dict_models[model].predict(X_strain)
            yhat = yhat.reshape(-1,1)
            yhat_strain.append(yhat)
            yhat = dict_models[model].predict(X_valid)
            yhat = yhat.reshape(-1,1)
            yhat_valid.append(yhat)
        else:
            y_strain_trans = label_encoder.transform(y_strain)
            dict_models[model].fit(X_strain, y_strain_trans)
            yhat = dict_models[model].predict(X_strain)
            yhat = label_encoder.inverse_transform(yhat)
            yhat = yhat.reshape(-1,1)
            yhat_strain.append(yhat)
            yhat = dict_models[model].predict(X_valid)
            yhat = label_encoder.inverse_transform(yhat)
            yhat = yhat.reshape(-1,1)
            yhat_valid.append(yhat)
    meta_X_strain = np.hstack(yhat_strain)
    meta_X_valid= np.hstack(yhat_valid)
    meta_X_strain_trans = one_hot_encode(meta_X_strain)
    meta_X_valid_trans = one_hot_encode(meta_X_valid)

    mod_ensemble.fit(meta_X_strain_trans, y_strain_trans)
    ensemble_pred = mod_ensemble.predict(meta_X_valid_trans)
    ensemble_pred = label_encoder.inverse_transform(ensemble_pred)
    ensemble_pred = ensemble_pred.reshape(-1,)

    super_f1.append(f1_score(y_valid, ensemble_pred, average="weighted"))
    print("superlearner_f1: " + str(super_f1[-1]))
    return super_f1

def ensemble_predict(X_test):
    yhat_list = []
    for model in dict_models:
        if model != "xgb":
            yhat_list.append(dict_models[model].predict(X_test).reshape(-1, 1))
        else:
            yhat = dict_models[model].predict(X_test)
            yhat = label_encoder.inverse_transform(yhat)
            yhat_list.append(yhat.reshape(-1, 1))

    meta_X_test = np.hstack(yhat_list)
    meta_X_test = one_hot_encode(meta_X_test)
    y_pred = mod_ensemble.predict(meta_X_test)
    return y_pred

def one_hot_encode(X_array):
    encoded_X_array = None
    for i in range(0, X_array.shape[1]):
        feature = label_encoder.transform(X_array[:,i])
        feature = feature.reshape(-1, 1)
        feature = onehot_encoder.transform(feature)
        if i==0:
            encoded_X_array = feature
        else:
            encoded_X_array = np.concatenate((encoded_X_array, feature), axis=1)
    return encoded_X_array

def cv_with_arrays(df_ml_to_array, df_cv_to_array, val_splits, run_list):
    best_algorithms = []
    array_ml, array_cv = df_ml_to_array.values, df_cv_to_array.values
    xgb_f1, xgb_params = [], []
    lasso_r2, lasso_params, lasso_nrmse = [], [], []
    rf_f1, rf_params = [], []
    svc_f1, svc_params = [], []
    super_f1 = []
    out_f1 = pd.DataFrame()
    df_f1 = pd.DataFrame()
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
            if "rf" in run_list: rf_f1, rf_params = find_params_rf(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain, rf_params=rf_params, rf_f1=rf_f1)
            if "svc" in run_list: svc_f1, svc_params = find_params_svc(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain, svc_params=svc_params, svc_f1=svc_f1)
            if "xgb" in run_list: xgb_f1, xgb_params = find_params_xgb(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain, xgb_params=xgb_params, xgb_f1=xgb_f1)
            if "super" in run_list: super_f1 = SuperLearner_fun(X_valid=X_valid, X_strain=X_strain, y_valid=y_valid, y_strain=y_strain, super_f1=super_f1, xgb_params=xgb_params, rf_params=rf_params, svc_params=svc_params)

        f1s = {"xgb": xgb_f1[-val_splits:], "rf": rf_f1[-val_splits:],
               "svc": svc_f1[-val_splits:], "super": super_f1[-val_splits:]}
        mean_f1 = {"xgb": np.mean(xgb_f1[-val_splits:]), "rf": np.mean(rf_f1[-val_splits:]),
               "svc": np.mean(svc_f1[-val_splits:]), "super": np.mean(super_f1[-val_splits:])}
        print(mean_f1)
        best_algorithms.append(sorted(mean_f1, key=lambda key: mean_f1[key])[-1])

        for model_name in run_list:  # Alle R2s und nrmses sammeln in jeweils einem df
            out_f1[model_name] = f1s[model_name]
            print(model_name + " median f1: " + str(median(out_f1[model_name])))

        for model_name in mean_f1:  # Alle R2s und nrmses sammeln in jeweils einem df
            df_f1[model_name].append(mean_f1[model_name])

    all_params = {"xgb": xgb_params, "rf": rf_params, "svc": svc_params}


    return df_f1, all_params, xgb_params, rf_params, svc_params, best_algorithms


# xgboost model
xgb_params_grid = {'max_depth': [2],
                   'learning_rate': [0.03],
                   'n_estimators': [1000],
                   'subsample': [0.4],
                   'colsample_bylevel': [0.1],
                   'colsample_bytree': [0.1]}  # Muss evtl. weg!'early_stopping_rounds': [500]


xgbc = xgboost.XGBClassifier(seed=20, objective="multi:softmax", num_class=4)

mod_xgb = GridSearchCV(estimator=xgbc, param_grid=xgb_params_grid, scoring='f1_weighted', verbose=0, n_jobs=-1, cv=5)


label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(["angst_depr", "depr_only", "angst_only", "andere"])


# Random Forest Model
rf_params_grid = {
    'bootstrap': [True],
    'max_depth': [2, 5, 8],
    'max_features': [2, 5],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [10, 20],
    'n_estimators': [1000]
}

rf = RandomForestClassifier(random_state=42)

mod_rf = GridSearchCV(estimator=rf, param_grid=rf_params_grid, scoring="f1_weighted", cv=5, n_jobs=-1, verbose=0)

# SVM

svc_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Auch hier müssen wir zuerst skalieren
    ('model', SVC())
])


svc_params_grid = {
    'model__kernel': ['rbf'],
    'model__C': [1, 2],  # hatte 1 oder 2 als Optimum
    'model__degree': [2, 3],
    'model__coef0': [0.00001, 0.00003, 0.0001],  # hatte 0.001 als Optimum
    'model__gamma': ['auto', 'scale']}

mod_svc = GridSearchCV(estimator=svc_pipeline, param_grid=svc_params_grid, scoring="f1_weighted", cv=5, n_jobs=-1, verbose=0)

# Ensemble:
xgb_ensemble_grid = {'max_depth': [2],
                   'learning_rate': [0.03],
                   'n_estimators': [1000],
                   'subsample': [0.4],
                   'colsample_bylevel': [0.1],
                   'colsample_bytree': [0.1]}



mod_ensemble = GridSearchCV(estimator=xgbc, param_grid=xgb_ensemble_grid, scoring='f1_weighted', n_jobs=-1, cv=5)

onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
onehot_encoder = onehot_encoder.fit([[0],[1], [2], [3]])

# Bei fertigem Datansatz AB HIER!!! --------------------------------------------------------------------------------------------------
outcome = "diagnose"  # Alternativ "srs_ges"
path = os.path.join(base_path, sub_folder_Patient)
os.chdir(path)
print(path)
df_ml = pd.read_excel('diagnosen.xlsx')
df_nested_cv = pd.read_excel('CV_folds_diagnosen.xlsx')
run_list = ["rf", "xgb", "svc", "super"]  # additional "rf", "bart", "lasso", "cnn", "xgb", "super", "cnn"; super geht nur wenn alle anderen drin sind.
val_sets = 5

df_f1, all_params, xgb_params, rf_params, svc_params, best_algorithms = cv_with_arrays(
    df_ml_to_array=df_ml, df_cv_to_array=df_nested_cv,
    val_splits=val_sets, run_list=run_list)


median(df_f1["xgb"])
median(df_f1["super"])
median(df_f1["rf"])
median(df_f1["svc"])

###############Ab hier gibt es Output, der gespeichert wird!!!!#################################################
path = os.path.join(base_path, sub_folder_output)
os.chdir(path)
print(path)
df_f1.to_excel('F1-inner-fold.xlsx')


#------------------------------ Hyperparameter Tuning und finales Modell
def list_params(params_dict):
    params_dict = {md: [params_dict[md]] for md in params_dict}
    return params_dict

def calculate_fit():
    y_pred = results.best_estimator_.predict(X_test_a)
    print(f1_score(y_test_a, y_pred, average="weighted"))

# encode string input values as integers

last_feature = df_ml.columns.tolist()[-2]
shaps = []
f1_list = []

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

    if best_algorithms[i]=="svc" or best_algorithms[i]=="super":
        super_svc = GridSearchCV(estimator=svc_pipeline, param_grid=svc_params_grid, scoring="f1_weighted", cv=5, n_jobs=-1, verbose=0)
        results = super_svc.fit(X_train_a, y_train_a)
        print(super_svc.best_params_)
        svc_pars = list_params(results.best_params_)
        mod_svc = GridSearchCV(estimator=svc_pipeline, param_grid=svc_pars, scoring="f1_weighted",
                               cv=5, n_jobs=-1, verbose=0)
        dict_models["svc"]=mod_svc

    '''
    if best_algorithms[i]=="lasso" or best_algorithms[i]=="super":
        super_lasso = GridSearchCV(lasso_pipeline, {"model__alpha": np.arange(0.01, 0.5, 0.005)}, cv=5, scoring="neg_mean_squared_error", verbose=0, n_jobs=-1)
        results = super_lasso.fit(X_train_a, y_train_a)
        print(super_lasso.best_params_)
        lasso_pars = list_params(results.best_params_)
        mod_lasso = GridSearchCV(lasso_pipeline, param_grid=lasso_pars, cv=5, scoring="neg_mean_squared_error",
                                 verbose=0, n_jobs=-1)
        dict_models["lasso"] = mod_lasso
    '''
    if best_algorithms[i]=="rf" or best_algorithms[i]=="super":
        super_rf = GridSearchCV(estimator=rf, param_grid=rf_params_grid, scoring="f1_weighted", cv=5, n_jobs=-1, verbose=0)
        results = super_rf.fit(X_train_a, y_train_a)
        print(super_rf.best_params_)
        rf_pars = list_params(results.best_params_)
        mod_rf = GridSearchCV(estimator=rf, param_grid=rf_pars, scoring="f1_weighted", cv=5, n_jobs=-1,
                              verbose=0)
        dict_models["rf"] = mod_rf

    if best_algorithms[i]=="xgb" or best_algorithms[i]=="super":
        xgb_params_grid = {'max_depth': [2], 'learning_rate': [0.03], 'n_estimators': [1000], 'subsample': [0.4], 'colsample_bylevel': [0.1], 'colsample_bytree': [0.1]}
        max_depths = list(range(2, 4))
        xgb_params_grid["max_depth"] = max_depths
        super_xgb = GridSearchCV(estimator=xgbc, param_grid=xgb_params_grid, scoring='f1_weighted', verbose=0, n_jobs=-1, cv=2)

        y_train_a_trans = label_encoder.transform(y_train_a)
        results = super_xgb.fit(X_train_a, y_train_a_trans)
        print(super_xgb.best_params_)

        subsamples = np.linspace(0.1, 1, 3)
        colsample_bytrees = np.linspace(0.1, 0.5, 3)
        colsample_bylevel = np.linspace(0.1, 0.5, 3)

        # merge into full param dicts
        params_dict = xgb_params_grid
        params_dict["subsample"] = subsamples
        params_dict["colsample_bytree"] = colsample_bytrees
        params_dict["colsample_bylevel"] = colsample_bylevel
        super_xgb = GridSearchCV(estimator=xgbc, param_grid=params_dict, scoring='f1_weighted', verbose=0, n_jobs=-1, cv=5)

        results = super_xgb.fit(X_train_a, y_train_a_trans, verbose=0)
        print(super_xgb.best_params_)
        learning_rates = np.logspace(-3, -0.7, 3)
        params_dict = list_params(results.best_params_)
        params_dict["learning_rate"] = learning_rates

        super_xgb = GridSearchCV(estimator=xgbc, param_grid=params_dict, scoring='f1_weighted', verbose=0, n_jobs=-1, cv=5)
        results = super_xgb.fit(X_train_a, y_train_a_trans, verbose=0)
        print(results.best_params_)
        xgb_pars = list_params(results.best_params_)
        mod_xgb = GridSearchCV(estimator=xgbc, param_grid=xgb_pars, scoring='f1_weighted', verbose=0, n_jobs=-1, cv=5)
        dict_models["xgb"] = mod_xgb

    yhat_train, yhat_test = [], []
    for model in dict_models:
        if model != "xgb":
            dict_models[model].fit(X_train_a, y_train_a)
            yhat = dict_models[model].predict(X_train_a)
            yhat = yhat.reshape(-1,1)
            yhat_train.append(yhat)
            yhat = dict_models[model].predict(X_test_a)
            yhat = yhat.reshape(-1,1)
            yhat_test.append(yhat)
        else:
            dict_models[model].fit(X_train_a, y_train_a_trans)
            yhat = dict_models[model].predict(X_train_a)
            yhat = label_encoder.inverse_transform(yhat)
            yhat = yhat.reshape(-1,1)
            yhat_train.append(yhat)
            yhat = dict_models[model].predict(X_test_a)
            yhat = label_encoder.inverse_transform(yhat)
            yhat = yhat.reshape(-1,1)
            yhat_test.append(yhat)


    if best_algorithms[i]=="super":
        meta_X_train = np.hstack(yhat_train)
        meta_X_test= np.hstack(yhat_test)
        # encode string input values as integers
        meta_X_train = one_hot_encode(meta_X_train)
        mod_ensemble.fit(meta_X_train, y_train_a_trans)

        meta_X_test = one_hot_encode(meta_X_test)
        pred = mod_ensemble.predict(meta_X_test)
        pred = label_encoder.inverse_transform(pred)
        pred = pred.reshape(-1, )
    else:
        pred = yhat_test[0]
        pred = pred.reshape(-1, )

    f1_list.append(f1_score(y_test_a, pred, average="weighted"))
    print("Auswahl: " + best_algorithms[i] + ": " + str(f1_score(y_test_a, pred, average="weighted")))

    if best_algorithms[i]=="super": explainer = shap.explainers.Permutation(ensemble_predict, masker=shap.sample(X_train_a, 100), max_evals=501)
    else: explainer = shap.explainers.Permutation(label_encoder.transform(dict_models[best_algorithms[i]].predict), masker=shap.sample(X_train_a, 100), max_evals=501)
    shaps.append(explainer(X_test_a))

# Save R2s and Nrmses
df_results = pd.DataFrame(
    {"f1": f1_list, "learner": best_algorithms})
df_results.to_excel('Results.docx')

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

shap.plots.bar(shap_values)

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
outcome = "diagnose"  # Alternativ "srs_ges"

df_ml, df_nested_cv = split_preparation(test_splits=test_sets, val_splits=val_sets, df=df,
                                        outcome=outcome)  # Alternativ "srs_ges"
df_ml = df_ml.iloc[:, 1:]  # erste Spalte löschen (session-Variable ist nicht ml-geeignet)

df_ml.to_excel("diagnosen.xlsx", index=False)
df_nested_cv.to_excel("CV_folds_diagnosen.xlsx", index=False)

