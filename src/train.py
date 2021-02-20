import pandas as pd
import numpy as np
from .models import XGBEnsemble
import joblib

if __name__ == "__main__":
    data = joblib.load("auto-insurance-fall-2017/train_folds.pkl")
    params = {
        "subsample": 0.8,
        "n_estimators": 400,
        "min_child_weight": 10,
        "max_depth": 4,
        "learning_rate": 0.05,
        "gamma": 5,
        "eta": 0.1,
        "colsample_bytree": 0.6,
        "eval_metric": "auc",
        "objective": "binary:logistic",
    }
    model = XGBEnsemble()
    oof = model.train(data, params)
    model.save("models/xgb")
