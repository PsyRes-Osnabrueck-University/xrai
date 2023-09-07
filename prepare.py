from sklearn.model_selection import RepeatedKFold, GroupKFold
import pandas as pd

def split_preparation(test_splits, val_splits, df, outcome, outcome_list, outcome_to_features, classed_splits=False):
    y = df[[outcome]]  # Outcome auswählen
    df_outcome_to_features = df.loc[:,outcome_to_features]
    df = df.drop(outcome_list, axis=1)
    df = pd.concat((df, df_outcome_to_features), axis=1)
    df[outcome] = y
    df = df.dropna()  # Missings fliegen raus!

    if not classed_splits:
        test_kf = RepeatedKFold(n_splits=test_splits, n_repeats=1, random_state=42)
        val_kf = RepeatedKFold(n_splits=val_splits, n_repeats=1, random_state=42)
    else:
        test_kf = GroupKFold(n_splits=test_splits)
        val_kf = GroupKFold(n_splits=val_splits)

    for outer_fold in range(test_splits):  # hinten dran kommt eine Variable für die folds. Darin steht in jedem Fold, wann man valid-set ist.
        df["fold_" + str(outer_fold)] = -1

    for i in range(len(df)):
        df["ID"] = df["Class"] + "_" + df["session"]
    columns = df.columns.tolist()
    a_data = df.values
    print(a_data)
    print(test_kf.split(a_data))

    if not classed_splits:
        for outer_fold, (train_index, test_index) in enumerate(test_kf.split(a_data)):  #
            a_train, a_test = a_data[train_index], a_data[test_index]
            print(outer_fold)
            inner_fold = 0
            for strain_index, valid_index in val_kf.split(a_train):
                print(inner_fold)
                a_strain, a_valid = a_train[strain_index], a_train[valid_index]
                df_valid = pd.DataFrame(a_valid, columns=columns)
                df.loc[df['ID'].isin(df_valid["ID"]), "fold_" + str(outer_fold)] = inner_fold
                # folds benennen, soweit eine row im valid-set ist (session und Class müssen stimmen)
                inner_fold += 1
    else:
        for outer_fold, (train_index, test_index) in enumerate(test_kf.split(a_data, groups=df["Class"])):  #
            a_train, a_test = a_data[train_index], a_data[test_index]
            print(outer_fold)
            inner_fold = 0
            for strain_index, valid_index in val_kf.split(a_train, groups=a_train[:, 0]):
                print(inner_fold)
                a_strain, a_valid = a_train[strain_index], a_train[valid_index]
                df_valid = pd.DataFrame(a_valid, columns=columns)
                df.loc[df['ID'].isin(df_valid["ID"]), "fold_" + str(outer_fold)] = inner_fold
                # folds benennen, soweit eine row im valid-set ist (session und Class müssen stimmen)
                inner_fold += 1

    df = df.drop("ID", axis = 1)
    df_cv = df.loc[:, "fold_0":]
    df_ml = df.loc[:, df.columns[2]:outcome]
    df_id = df.iloc[:, 0:2]
    return df_id, df_ml, df_cv

print("Prepare is imported!")