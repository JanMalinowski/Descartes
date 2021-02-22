from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import joblib


def create_folds(df: pd.DataFrame, y: np.array, n_splits: int = 5) -> pd.DataFrame:
    """
    Function for creating stratified k fold validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    df["kfold"] = -1

    for k, (train_index, test_index) in enumerate(skf.split(df.values, y)):
        df.loc[test_index, "kfold"] = k
        print(f"Fold no.{k}, Number of datapoints in the fold {len(train_index)}")

    return df


if __name__ == "__main__":
    df = joblib.load("auto-insurance-fall-2017/df_all.pkl")
    target = joblib.load("auto-insurance-fall-2017/target.pkl")

    train_idxs = df["is_train"] == 1
    test_idxs = df["is_train"] == 0

    train = df.loc[train_idxs, :]
    train.loc[:, "target"] = target
    train = train.sample(frac=1.0).reset_index(drop=True)

    test = df.loc[test_idxs, :]
    train = train.drop(["is_train"], axis=1)
    test = test.drop(["is_train"], axis=1)

    print("Creating stratified folds!")
    train = create_folds(df=train, y=target, n_splits=5)
    print("Finished creating folds!")

    joblib.dump(train, "auto-insurance-fall-2017/train_folds.pkl")
    joblib.dump(test, "auto-insurance-fall-2017/test.pkl")
