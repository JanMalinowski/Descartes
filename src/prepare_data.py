import pandas as pd
from pathlib import Path
from .utils import change_money
from .categorical import Encoder
import joblib


def prepare_data(data_dir: Path = Path("auto-insurance-fall-2017")) -> None:
    """
    This function wraps-up the data preprocessing process from the jupyter notebook.
    Returns:
    - target.pkl - target flag for the train set
    - target_amt.pkl - target amount for the train set
    - df_all.pkl - preprocessed dataframe with train and test data. They can be differentiated
      by looking at the is_train column.
    """
    train_df = pd.read_csv(data_dir / "train_auto.csv")
    test_df = pd.read_csv(data_dir / "test_auto.csv")

    money_cols = ["INCOME", "HOME_VAL", "BLUEBOOK", "OLDCLAIM"]
    train_df = change_money(train_df, money_cols)
    test_df = change_money(test_df, money_cols)

    for col in ["AGE", "YOJ", "CAR_AGE"]:
        train_df[col] = train_df[col].astype(float)
        test_df[col] = test_df[col].astype(float)

    numeric_cols = train_df.columns[train_df.dtypes != "object"]
    nonnumeric_cols = train_df.columns[train_df.dtypes == "object"]

    target = train_df["TARGET_FLAG"]
    target_amt = train_df["TARGET_AMT"]

    train_df = train_df.drop(["TARGET_FLAG", "TARGET_AMT"], axis=1)
    test_df = test_df.drop(["TARGET_FLAG", "TARGET_AMT"], axis=1)

    train_df["is_train"] = 1
    test_df["is_train"] = 0

    df = pd.concat([train_df, test_df]).reset_index(drop=True)

    df[nonnumeric_cols] = df[nonnumeric_cols].fillna("-99999")
    df = df.fillna(-99999)

    # Label encoding the data
    enc = Encoder(nonnumeric_cols)
    df = enc.fit_transform(df)
    enc.save(path="production_tools/encoders.pkl")

    joblib.dump(target, data_dir / "target.pkl")
    joblib.dump(target_amt, data_dir / "target_amt.pkl")
    joblib.dump(df, data_dir / "df_all.pkl")


if __name__ == "__main__":
    print("Starting to preprocessing the data!")
    prepare_data()
    print("Finished preprocessing the data!")
