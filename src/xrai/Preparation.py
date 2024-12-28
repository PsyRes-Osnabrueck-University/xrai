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

from xrai.utils import split_preparation, identify_last_column
from xrai.cv_fold import cv_with_arrays


class Preparation:

    def __init__(
        self,
        base_path=os.getcwd(),
        file_name="sample.xlsx",
        outcome=None,
        outcome_list=[],
        classed_splits=False,
        outcome_to_features=[],
        test_sets=10,
        val_sets=5,
    ) -> None:

        self.base_path = base_path
        self.file_name = file_name
        self.outcome = outcome
        self.outcome_list = outcome_list
        self.classed_splits = classed_splits
        self.outcome_to_features = outcome_to_features
        self.test_sets = test_sets
        self.val_sets = val_sets

        self.output_path = os.path.join(self.base_path, "output")

        os.makedirs(self.output_path, exist_ok=True)
        os.chdir(self.base_path)
        print(self.base_path)

        self.df = pd.read_excel(self.file_name)

        if self.outcome is None:
            self.outcome = identify_last_column(self.df)

        # create prepared ml dataset
        self.df_id, self.df_ml, self.df_nested_cv = split_preparation(
            test_splits=self.test_sets,
            val_splits=self.val_sets,
            df=self.df,
            outcome=self.outcome,
            outcome_list=self.outcome_list,
            outcome_to_features=self.outcome_to_features,
            classed_splits=self.classed_splits,
        )

        pass

    def ensemble_predict(X_test):
        fold = X_test[:, 0]
        change_id = np.where(fold[1:] != fold[:-1])[0] + 1
        block_start_indices = [0] + change_id.tolist() + [len(fold)]
        block_arrays = []
        for i in range(len(block_start_indices) - 1):
            start = block_start_indices[i]
            end = block_start_indices[i + 1]
            block_values = X_test[start:end, :]
            block_arrays.append(block_values)
        y_list = []
        for idx, X in enumerate(block_arrays):
            yhat_list = []
            index = round(float(fold[block_start_indices[idx]]))
            group_test = X[:, 1]
            X = X[:, 3:]
            for model in dict_models[index]:
                if model != "gpb" and model != "merf" and model != "super":
                    yhat_list.append(
                        dict_models[index][model].predict(X).reshape(-1, 1)
                    )
                elif model == "gpb":
                    pred = dict_models[index][model].predict(
                        data=np.delete(X, Z_loc, axis=1),
                        group_data_pred=group_test,
                        group_rand_coef_data_pred=X[:, Z_loc],
                        predict_var=True,
                        pred_latent=False,
                    )
                    yhat = pred["response_mean"].reshape(-1, 1)
                    yhat_list.append(yhat)
                elif model == "merf":
                    z = np.array([1] * len(X)).reshape(-1, 1)
                    z = np.hstack([z, X[:, Z_loc]]).astype(float)
                    group_test = pd.Series(
                        group_test.reshape(
                            -1,
                        )
                    )
                    yhat = dict_models[index][model].predict(
                        X=np.delete(X, Z_loc, axis=1), Z=z, clusters=group_test
                    )
                    yhat_list.append(yhat.reshape(-1, 1))
            meta_X_test = np.hstack(yhat_list)
            y_pred = dict_models[index]["super"].predict(meta_X_test)
            y_list.append(y_pred.reshape(-1, 1))
        y = np.array(np.concatenate(y_list, axis=0))
        return y

    def create_folds(
        self,
        Z_list=[],
        # Select the compething algorithms
        run_list=["merf", "super", "gpb"],
        val_sets=5,
        feature_selection=True,  # Should feature selection be conducted?
        xAI=True,
    ):

        self.feature_selection = feature_selection
        self.xAI = xAI
        self.Z_list = Z_list
        self.run_list = run_list

        output_dir = self.output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        os.chdir(output_dir)

        # Save the prepared data in an excel with multiple sheets in the out_folder.
        output_file_name = f"processed_output_{self.outcome}.xlsx"
        output_file_path = os.path.join(output_dir, output_file_name)

        writer = pd.ExcelWriter(output_file_path, engine="xlsxwriter")
        # Write each dataframe to a different worksheet.
        self.df_id.to_excel(writer, sheet_name="ID", index=False)
        self.df_ml.to_excel(writer, sheet_name="ML", index=False)
        self.df_nested_cv.to_excel(writer, sheet_name="CV", index=False)
        writer.close()

        sub_folder_output = os.path.join(output_dir, "Out_Supervision")
        if not os.path.exists(sub_folder_output):
            os.makedirs(sub_folder_output)
        os.chdir(sub_folder_output)

        val_sets = len(set(self.df_nested_cv["fold_0"])) - 1

        df_r, df_nrmse, self.all_params, self.best_algorithms = cv_with_arrays(
            df_ml=self.df_ml,
            df_cv=self.df_nested_cv,
            val_splits=val_sets,
            run_list=run_list,
            feature_selection=feature_selection,
            series_group=self.df_id["Class"],
            classed=self.classed_splits,
            random_effects=Z_list,
        )

        df_r.to_excel("r-inner-fold.xlsx")
        df_nrmse.to_excel("Nrmse-inner-fold.xlsx")

        return True
