import xgboost as xgb
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score



class XGBEnsemble:
    def __init__(self):
        self.models = []
    
    def train(self, data: pd.DataFrame, params: dict, n_splits: int=5)-> np.array:
        cols = [col for col in data.columns if col not in ["target", "kfold", "INDEX"]]
        oof = np.zeros(data["target"].shape[0])
        
        for k in range(n_splits):
            print(50*"=")
            print(f"Fold {k}")
            train_idxs = (data["kfold"]!=k)
            val_idxs = (data["kfold"]==k)
            
            dtrain = xgb.DMatrix(data.loc[train_idxs, cols], label=data.loc[train_idxs, "target"])
            dval = xgb.DMatrix(data.loc[val_idxs, cols], label=data.loc[val_idxs, "target"])
            evallist = [(dtrain, 'train'), (dval, 'validation')]
            bst = xgb.train(params, dtrain, evals=evallist)
            oof[val_idxs] = bst.predict(dval, ntree_limit=bst.best_ntree_limit+50)
            self.models.append(bst)
            
        auc = roc_auc_score(data["target"], oof)
        print(f"OOF ROC AUC is: {auc}")
        
        return oof
    
    def predict(self, data: pd.DataFrame) -> np.array:
        if not self.models:
            raise "Fit model first"
        
        preds = np.zeros(data.shape[0])
        for i, model in enumerate(self.models):
            dtest = xgb.DMatrix(data[model.feature_names])
            preds += model.predict(dtest,
                    ntree_limit=model.best_ntree_limit+50)/len(self.models)
        return preds
            
    def save(self, path: Path) -> None:
        for i, model in enumerate(self.models):
            joblib.dump(model, 
                        os.path.join(path, f"xgb_model{i}.pkl"))
            
    def load(self, path: Path) -> None:
        models = os.listdir(path)
        for i, model in enumerate(models):
            self.models.append(joblib.load(
                                os.path.join(path, model)))

            