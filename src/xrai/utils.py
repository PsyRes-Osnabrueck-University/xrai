from featurewiz import FeatureWiz
from sklearn import metrics
import xgboost
import shap
import os
from merf import MERF
import scipy.stats
from sklearn.model_selection import RepeatedKFold, GroupKFold
import pandas as pd

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
# import BerTopic.src.xrai.transform as transform

import gpboost as gpb

from featurewiz import FeatureWiz
from sklearn import metrics
import xgboost
import shap
import os
from merf import MERF



def identify_last_column(df) -> str:
    """
    Identifies the last column in the Excel file.
    """
    last_column_header = df.columns[-1]
    return last_column_header


def split_preparation(test_splits,
                      val_splits,
                      df,
                      outcome,
                      outcome_list,
                      outcome_to_features,
                      classed_splits=False):

    y = df[[outcome]]  # Outcome auswählen
    df_outcome_to_features = df.loc[:, outcome_to_features]
    df = df.drop(outcome_list, axis=1)
    df = pd.concat((df, df_outcome_to_features), axis=1)
    df[outcome] = y
    df = df.dropna()  # Missings fliegen raus!
    df = df.reset_index(drop=True)
    if not classed_splits:
        test_kf = RepeatedKFold(n_splits=test_splits,
                                n_repeats=1, random_state=42)
        val_kf = RepeatedKFold(n_splits=val_splits,
                               n_repeats=1, random_state=42)
    else:
        test_kf = GroupKFold(n_splits=test_splits)
        val_kf = GroupKFold(n_splits=val_splits)

    # hinten dran kommt eine Variable für die folds. Darin steht in jedem Fold, wann man valid-set ist.
    for outer_fold in range(test_splits):
        df["fold_" + str(outer_fold)] = -1
    df["ID"] = "id"
    for i in range(len(df)):
        df["ID"][i] = str(df["Class"][i]) + "_" + str(df["session"][i])
    columns = df.columns.tolist()
    a_data = df.values
    # print(df["ID"])

    if not classed_splits:
        for outer_fold, (train_index, test_index) in enumerate(test_kf.split(a_data)):  #
            a_train, a_test = a_data[train_index], a_data[test_index]
            # print(outer_fold)
            inner_fold = 0
            for strain_index, valid_index in val_kf.split(a_train):
                # print(inner_fold)
                a_strain, a_valid = a_train[strain_index], a_train[valid_index]
                df_valid = pd.DataFrame(a_valid, columns=columns)
                df.loc[df['ID'].isin(df_valid["ID"]),
                       "fold_" + str(outer_fold)] = inner_fold
                # folds benennen, soweit eine row im valid-set ist (session und Class müssen stimmen)
                inner_fold += 1
    else:
        #
        for outer_fold, (train_index, test_index) in enumerate(test_kf.split(a_data, groups=df["Class"])):
            a_train, a_test = a_data[train_index], a_data[test_index]
            # print(outer_fold)
            inner_fold = 0
            for strain_index, valid_index in val_kf.split(a_train, groups=a_train[:, 0]):
                # print(inner_fold)
                a_strain, a_valid = a_train[strain_index], a_train[valid_index]
                df_valid = pd.DataFrame(a_valid, columns=columns)
                df.loc[df['ID'].isin(df_valid["ID"]),
                       "fold_" + str(outer_fold)] = inner_fold
                # folds benennen, soweit eine row im valid-set ist (session und Class müssen stimmen)
                inner_fold += 1

    df = df.drop("ID", axis=1)
    df_cv = df.loc[:, "fold_0":]
    df_ml = df.loc[:, df.columns[2]:outcome]
    df_id = df.iloc[:, 0:2]
    return df_id, df_ml, df_cv


# print("Prepare is imported!")

def list_params(params_dict):
    params_dict = {md: [params_dict[md]] for md in params_dict}
    return params_dict

def cor(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        x, y)
    return r_value

def custom_masker(mask, x):
    mean_mask = shap.sample(X_train_a, 200).mean(axis=0)
    mean_mask = np.insert(mean_mask, 0, [1, 1, 1])
    mask2 = (mean_mask * mask).reshape(1, len(mask))
    fold, group, session = x[0], x[1], x[2]
    x[1], x[2] = 0, 0
    x = x.astype(float)
    out = (x * mask2).reshape(1, len(x))
    out = out.astype(object)
    out[0, 0], out[0, 1], out[0, 2] = fold, group, session
    return out  # in this simple example we just zero out the features we are masking

def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)

def weighted_std(data, weights):
    weighted_mean_val = weighted_mean(data, weights)
    squared_diff = np.sum(weights * (data - weighted_mean_val)**2)
    return np.sqrt(squared_diff / np.sum(weights))

# normal is no limits, cor is [-1, 1], nrmse = [0, ], f1 = [0, 1]
def mean_ci(sample, weights, limits="normal"):
    mean = weighted_mean(sample, weights)
    sd = weighted_std(sample, weights)
    if limits == "bootstrap":
        r = 1000
        probabilities = weights / weights.sum()
        monte_boot_mean = [np.mean(np.random.choice(np.array(sample), len(
            sample), probabilities.tolist())) for _ in range(r)]
        mean_out = np.mean(monte_boot_mean)
        lower = np.percentile(monte_boot_mean, 2.5)
        upper = np.percentile(monte_boot_mean, 97.5)
    if limits == "normal":
        mean_out = mean
        lower = mean-1.96*sd
        upper = mean+1.96*sd
    if limits == "cor":
        mean_trans = (mean+1)/2
        if mean_trans == 0:
            mean_trans = 1e-7
        try:
            L = math.log(mean_trans/(1-mean_trans))
        except:
            L = 1e7
        sd_L = sd/(mean_trans*(1-mean_trans))
        mean_out = math.exp(L)/(math.exp(L)+1)*2-1
        lower = math.exp(L-1.96*sd_L*0.5)/(math.exp(L-1.96*sd_L*0.5)+1)*2-1
        upper = math.exp(L+1.96*sd_L*0.5)/(math.exp(L+1.96*sd_L*0.5)+1)*2-1
    if limits == "nrmse":

        try:
            L = math.log(mean)
        except:
            L = -1e7
        sd_L = sd/mean
        mean_out = mean
        lower = math.exp(L - 1.96 * sd_L)
        upper = math.exp(L + 1.96 * sd_L)
    if limits == "f1":
        if mean == 0:
            mean = 1e-7
        try:
            L = math.log(mean/(1-mean))
        except:
            L = 1e7
        sd_L = sd/(mean*(1-mean))
        mean_out = math.exp(L) / (math.exp(L) + 1)
        lower = math.exp(L - 1.96 * sd_L) / (math.exp(L - 1.96 * sd_L) + 1)
        upper = math.exp(L + 1.96 * sd_L) / (math.exp(L + 1.96 * sd_L) + 1)
    return mean_out, lower, upper
