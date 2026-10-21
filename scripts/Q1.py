"""思考题1: coefficient paths 复现"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import src.models as models
import src.datasets as ds
import numpy as np
import tqdm
import os

from configs import Q1_Configs as configs

np.random.seed(configs.SEED)

if __name__ == "__main__":
    data = ds.diabetes()
    X, Y, Name = data['X'], data['Y'], data['Name']
    n_samples, n_features = X.shape

    os.makedirs(configs.SAVE_DIR, exist_ok = True)

    L1_lambda = np.linspace(configs.L1_MIN, configs.L1_MAX, configs.N_LAMBDAS)
    L2_lambda = np.linspace(configs.L2_MIN, configs.L2_MAX, configs.N_LAMBDAS)

    W_l1_log, W_l2_log = [], []
    for l1 in tqdm.tqdm(L1_lambda, desc = "L1"):
        model = models.LinearRegression(
            LearningRate = configs.LEARNING_RATE, 
            MaxIters = configs.MAX_ITERS, 
            Lambda1 = l1,
            Lambda2 = 0.0
        )
        model.fit(X, Y, use_prox_l1 = True)
        W_l1_log.append(model.Weights)

    for l2 in tqdm.tqdm(L2_lambda, desc = "L2"):
        model = models.LinearRegression(
            LearningRate = configs.LEARNING_RATE, 
            MaxIters = configs.MAX_ITERS, 
            Lambda1 = 0.0,
            Lambda2 = l2
        )
        model.fit(X, Y)
        W_l2_log.append(model.Weights)

    plt.figure()
    for i in range(n_features):
        plt.plot(L1_lambda, [w[i] for w in W_l1_log], '-o', label = Name[i])
    plt.xlabel('L1')
    plt.ylabel('Coefficients')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(configs.SAVE_DIR, "Q1_coefficient_paths_L1.png"))
    plt.close()

    plt.figure()
    for i in range(n_features):
        plt.plot(L2_lambda, [w[i] for w in W_l2_log], '-o', label = Name[i])
    plt.xlabel('L2')
    plt.ylabel('Coefficients')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(configs.SAVE_DIR, "Q1_coefficient_paths_L2.png"))
    plt.close()