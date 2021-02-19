import xgboost as xgb
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from .models import XGBEnsemble

if __name__ == "__main__":
    data = joblib.load("auto-insurance-fall-2017/test.pkl")
    model = XGBEnsemble()
    model.load("models/xgb/")
    preds = model.predict(data)
    results = pd.DataFrame({"INDEX": data["INDEX"], "proba": preds})
    results.to_csv("auto-insurance-fall-2017/submission.csv", index=False)
