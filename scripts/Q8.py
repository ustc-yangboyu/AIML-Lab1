"""思考题8: truth 训练/验证集分布偏移"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition
import sklearn.metrics
import sklearn.model_selection

import src.datasets as ds
import src.models as models
import src.utils as utils
from configs import Q8_Configs as configs

np.random.seed(configs.SEED)

if __name__ == "__main__":
    Data = ds.load_truth(configs.TRUTH_DIR, configs.LAYER)
    X = np.vstack([Data['X_train'], Data['X_valid']])
    Y = np.r_[np.zeros(len(Data['X_train'])), np.ones(len(Data['X_valid']))]
    Xf, Xh, Yf, Yh = sklearn.model_selection.train_test_split(
        X, Y,
        test_size = configs.TEST_RATIO,
        stratify = Y,
        random_state = configs.SEED
    )

    Xfs, Mean, Std = utils.zscore(Xf)
    Xhs, _, _ = utils.zscore(Xh, Mean, Std)
    Cs = np.logspace(np.log10(configs.C_MIN), np.log10(configs.C_MAX), configs.N_CS)
    Folds = sklearn.model_selection.StratifiedKFold(configs.N_FOLDS, shuffle = True, random_state = configs.SEED)
    Cv, _, Fits = models.cv_search(
        Cs, lambda c: models.Sklearn_Logistic(C = c, ClassWeight = 'balanced'),
        Xfs, Yf, Folds, lambda y, s: 1.0 - sklearn.metrics.roc_auc_score(y, s)
    )
    i = int(np.argmin(Cv))
    Clf = Fits[i]
    Score = Clf.predict(Xhs)
    Pc1 = sklearn.decomposition.PCA(1).fit(Xfs).transform(Xhs)[:, 0]

    Stats = {
        'layer': configs.LAYER, 'C': float(Cs[i]), 'cv_auc': float(1.0 - Cv[i]),
        'holdout_ratio': configs.TEST_RATIO,
        'n_fit': len(Yf), 'n_holdout': len(Yh),
        'auc': float(sklearn.metrics.roc_auc_score(Yh, Score)),
        'balanced_acc': float(sklearn.metrics.balanced_accuracy_score(Yh, Score > 0)),
        'score_mean_train': float(Score[Yh == 0].mean()), 'score_mean_valid': float(Score[Yh == 1].mean()),
    }

    plt.figure(figsize = (7, 5))
    for k, Color, Label, Size in ((0, 'tab:blue', 'train', 15), (1, 'tab:red', 'valid', 30)):
        M = Yh == k
        plt.scatter(Score[M], Pc1[M], s = Size, alpha = 0.7, c = Color, label = f'{Label} (n = {M.sum()})')

    plt.axvline(0, color = 'gray', linestyle = '--', linewidth = 1)
    plt.xlabel('Projection on LR decision normal (domain score)')
    plt.ylabel('Projection on PC1 (fit set)')
    plt.title(f"Domain Separability (holdout {configs.TEST_RATIO:.0%}): auc = {Stats['auc']:.3f} (chance = 0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(configs.SAVE_DIR) / 'Q8_domain_shift.png')
    plt.close()

    with open(Path(configs.RESULTS_DIR) / 'Q8_domain_shift.json', 'w') as f:
        json.dump(Stats, f, indent = 2)
    print(json.dumps(Stats, indent = 2))
    print('Figure and Stats Saved.')
