from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
import pandas as pd

def split_preparation(test_splits, val_splits, df, outcome, outcome_list, outcome_to_features):
    y = df[[outcome]]  # Outcome auswählen
    df_outcome_to_features = df.loc[:,outcome_to_features]
    df = df.drop(outcome_list, axis=1)
    df = pd.concat([df, df_outcome_to_features], axis=1, join="inner")
    df[outcome] = y
    df = df.dropna()  # Missings fliegen raus!
    test_kf = RepeatedKFold(n_splits=test_splits, n_repeats=1, random_state=42)
    val_kf = RepeatedKFold(n_splits=val_splits, n_repeats=1, random_state=42)

    for outer_fold in range(test_splits):  # hinten dran kommt eine Variable für die folds. Darin steht in jedem Fold, wann man valid-set ist.
        df["fold_" + str(outer_fold)] = -1
    columns = df.columns.tolist()

    a_data = df.values
    print(test_kf.split(a_data))
    for outer_fold, (train_index, test_index) in enumerate(test_kf.split(a_data)):  #
        a_train, a_test = a_data[train_index], a_data[test_index]
        print(outer_fold)
        inner_fold = 0
        for strain_index, valid_index in val_kf.split(a_train):
            print(inner_fold)
            a_strain, a_valid = a_train[strain_index], a_train[valid_index]
            df_valid = pd.DataFrame(a_valid, columns=columns)
            session_list = df_valid["session"].tolist()
            Class_list = df_valid["Class"].tolist()
            df.loc[(df['session'].isin(session_list) & df['Class'].isin(Class_list)), "fold_" + str(outer_fold)] = inner_fold
            # folds benennen, soweit eine row im valid-set ist (session und Class müssen stimmen)
            inner_fold += 1

    df_cv = df.loc[:, "fold_0":]
    df_ml = df.loc[:, df.columns[2]:outcome]
    df_id = df.iloc[:, 0:2]
    return df_id, df_ml, df_cv
