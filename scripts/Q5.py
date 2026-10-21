"""思考题5: world 正则化选择与最终权重提交(Ridge/Lasso 各自 5 折 CV 搜 λ 并画图;配置取 Q4 验证最优;原始特征空间直接拟合)"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection

import src.datasets as ds
import src.models as models
import src.utils as utils
from configs import Q5_Configs as configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices = ('base', 'extra'), default = None, help = '默认取 Q4_layer_sweep.json 的全局最优')
    parser.add_argument('--layer', type = int, default = None, help = '默认同上')
    args = parser.parse_args()

    with open(Path(configs.RESULTS_DIR) / 'Q4_layer_sweep.json') as f:
        Sweep = json.load(f)
    Scores = {(v, int(l)): R[v]['hav_median_km'] for l, R in Sweep.items() for v in R}
    Version, Layer = min(Scores, key = Scores.get)
    Version = args.version or Version
    Layer = args.layer if args.layer is not None else Layer

    data = ds.load_world(Version, Layer)
    print(f"Config: {Version} L{Layer}(Q4 valid 最优 {Scores[(Version, Layer)]:.1f} km),5 折 CV 选 λ")

    Lambdas2 = np.logspace(np.log10(configs.L2_MIN), np.log10(configs.L2_MAX), configs.N_LAMBDAS)
    Lambdas1 = np.logspace(np.log10(configs.L1_MIN), np.log10(configs.L1_MAX), configs.N_LAMBDAS)
    Folds = sklearn.model_selection.KFold(configs.N_FOLDS, shuffle = True, random_state = configs.SEED)
    Metric = lambda Y, P: utils.eval_world(Y, P)['hav_median_km']

    print('Ridge (L2):')
    Ridge_Cv, Ridge_Va, Ridge_Fits = models.cv_search(
        Lambdas2, lambda l: models.Sklearn_Ridge(Lambda2 = l),
        data['X_train'], data['Y_train'], Folds, Metric,
        Xv = data['X_valid'], Yv = data['Y_valid'])
    print('Lasso (L1):')
    Lasso_Cv, Lasso_Va, Lasso_Fits = models.cv_search(
        Lambdas1, lambda l: models.Sklearn_Lasso(Lambda1 = l, MaxIters = configs.LASSO_MAX_ITERS),
        data['X_train'], data['Y_train'], Folds, Metric,
        Xv = data['X_valid'], Yv = data['Y_valid'])

    i2 = int(np.argmin(Ridge_Cv))
    M = Ridge_Fits[i2]
    np.savez(Path(configs.RESULTS_DIR) / 'world.npz', weights = M.coef_.astype(np.float64), bias = M.intercept_.astype(np.float64))

    i1 = int(np.argmin(Lasso_Cv))
    with open(Path(configs.RESULTS_DIR) / 'Q5_final_config.json', 'w') as f:
        json.dump(
            {
                'version': Version, 'layer': Layer,
                'lambda2': float(Lambdas2[i2]),
                'ridge': {
                    'lambdas': Lambdas2.tolist(),
                    'cv_median_km': Ridge_Cv,
                    'valid_median_km': Ridge_Va
                },
                'lasso': {
                    'lambdas': Lambdas1.tolist(),
                    'cv_median_km': Lasso_Cv,
                    'valid_median_km': Lasso_Va,
                    'nnz': [int(np.sum(m.coef_ != 0)) for m in Lasso_Fits]
                },
            },
            f, indent = 2
        )
    print(f"Ridge best : λ2 = {Lambdas2[i2]:7.3g}, cv {Ridge_Cv[i2]:7.1f} km, valid {Ridge_Va[i2]:7.1f} km")
    print(f"Lasso best : λ1 = {Lambdas1[i1]:7.3g}, cv {Lasso_Cv[i1]:7.1f} km, valid {Lasso_Va[i1]:7.1f} km, nnz {int(np.sum(Lasso_Fits[i1].coef_ != 0))}")

    fig, Axes = plt.subplots(1, 2, figsize = (12, 5), sharey = True)
    for Ax, Lambdas, Cv, Va, i, Xlabel, Title in (
        (Axes[0], Lambdas2, Ridge_Cv, Ridge_Va, i2, 'λ2', 'Ridge (L2)'),
        (Axes[1], Lambdas1, Lasso_Cv, Lasso_Va, i1, 'λ1', 'Lasso (L1)'),
    ):
        Ax.plot(Lambdas, Cv, '-o', markersize = 3, label = '5-fold CV (train)')
        Ax.plot(Lambdas, Va, '--s', markersize = 3, label = 'valid (reference)')
        Ax.scatter(Lambdas[i], Cv[i], marker = '*', s = 120, zorder = 5, color = 'tab:red', label = 'chosen by CV')
        Ax.set(xscale = 'log', xlabel = Xlabel, title = Title)
        Ax.legend(fontsize = 8)
        Ax.grid(alpha = 0.3)

    Axes[0].set_ylabel('Median Haversine Error (km)')
    fig.suptitle(f'World Probe λ Search ({Version} L{Layer})')
    plt.tight_layout()
    plt.savefig(Path(configs.SAVE_DIR) / 'Q5_lambda_sweep.png')
    plt.close()
    print('Figure Saved.')
