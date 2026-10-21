"""思考题9: truth 特征层与正则化选择"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

import src.datasets as ds
import src.models as models
import src.utils as utils
from configs import Q9_Configs as configs

if __name__ == "__main__":
    Cs = np.logspace(np.log10(configs.C_MIN), np.log10(configs.C_MAX), configs.N_CS)
    Folds = sklearn.model_selection.StratifiedKFold(configs.N_FOLDS, shuffle = True, random_state = configs.SEED)

    Results = {}
    for layer in range(configs.N_LAYERS):
        Data = ds.load_truth(configs.TRUTH_DIR, layer)
        Xs, _, _ = utils.zscore(Data['X_train'])
        ClfCV = sklearn.linear_model.LogisticRegressionCV(
            Cs = Cs, cv = Folds, scoring = 'accuracy', solver = 'lbfgs',
            max_iter = 2000, n_jobs = 4, refit = False)
        ClfCV.fit(Xs, Data['Y_train'])
        Results[layer] = ClfCV.scores_[1].mean(axis = 0).tolist()
        print(f"Layer {layer:2d} | cv acc {max(Results[layer]):.4f} "
              f"(C = {Cs[int(np.argmax(Results[layer]))]:.4g})", flush = True)

    Best_Layer = max(Results, key = lambda l: max(Results[l]))
    Best_C = float(Cs[int(np.argmax(Results[Best_Layer]))])

    Data = ds.load_truth(configs.TRUTH_DIR, Best_Layer)
    X, Y, Yv = Data['X_train'], Data['Y_train'], Data['Y_valid']
    Xs, Mean, Std = utils.zscore(X)

    Scores = np.zeros(len(X))
    for Tr, Te in Folds.split(Xs, Y):
        Clf = models.Sklearn_Logistic(C = Best_C)
        Clf.fit(Xs[Tr], Y[Tr])
        Scores[Te] = Clf.predict(Xs[Te])
    Shifts = np.quantile(Scores, np.linspace(0.01, 0.99, 197))
    Best_Shift = float(Shifts[int(np.argmax([utils.eval_acc(Scores - s, Y) for s in Shifts]))])

    Clf = models.Sklearn_Logistic(C = Best_C)
    Clf.fit(Xs, Y)
    W, B = models.to_raw(Clf.coef_[0], Clf.intercept_[0], Mean, Std)
    B = float(B - Best_Shift)
    Final = {
        'layer': Best_Layer, 'C': Best_C, 'threshold_shift': Best_Shift,
        'lambda2_course': 1 / (2 * Best_C * len(X)),
        'train_acc': utils.eval_acc(Clf.predict(Xs) - Best_Shift, Y),
        'valid_acc': utils.eval_acc(Data['X_valid'] @ W + B, Yv),
        'valid_auc': float(sklearn.metrics.roc_auc_score(Yv, Data['X_valid'] @ W + B))
    }
    print(json.dumps(Final, indent = 2))

    np.savez(Path(configs.RESULTS_DIR) / 'truth.npz', weights = np.asarray(W, np.float64), bias = np.asarray(B, np.float64))
    with open(Path(configs.RESULTS_DIR) / 'Q9_final_config.json', 'w') as f:
        json.dump({'final': Final, 'Cs': Cs.tolist(), 'per_layer': Results}, f, indent = 2)
