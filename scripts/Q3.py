"""思考题3: 与 sklearn 实现的拟合效果对比"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import src.models as models
import src.datasets as ds
import src.utils as utils
import numpy as np
import sklearn
import tqdm
import os

from configs import Q3_Configs as configs

np.random.seed(42)

if __name__ == '__main__':
    data = ds.diabetes()
    X, Y, Name = data['X'], data['Y'], data['Name']
    X_train, Y_train, X_test, Y_test = ds.split_datasets(X, Y, test_size = configs.TEST_RATIO).values()

    Lambda_1 = np.linspace(configs.L1_MIN, configs.L1_MAX, configs.N_LAMBDAS)
    Lambda_2 = np.linspace(configs.L2_MIN, configs.L2_MAX, configs.N_LAMBDAS)

    Test_Loss_Sklearn_L1, Test_Loss_Ours_L1 = [], []
    Test_Loss_Sklearn_L2, Test_Loss_Ours_L2 = [], []
    for l1 in tqdm.tqdm(Lambda_1, desc = 'L1'):
        Sklearn_LinearRegression_L1 = models.Sklearn_Lasso(
            Lambda1 = l1,
            MaxIters = configs.MAX_ITERS,
            WithNorm = True,
        )
        Ours_LinearRegression_L1 = models.LinearRegression(
            LearningRate = configs.LEARNING_RATE,
            MaxIters = configs.MAX_ITERS,
            Lambda1 = l1,
            Lambda2 = 0
        )
        Ours_LinearRegression_L1.fit(X_train, Y_train, use_prox_l1 = True)
        Test_Loss_Ours_L1.append(utils.Test(
            Model = Ours_LinearRegression_L1,
            X_test = X_test,
            Y_test = Y_test
        ))
        Sklearn_LinearRegression_L1.fit(X_train, Y_train)
        Test_Loss_Sklearn_L1.append(utils.Test(
            Model = Sklearn_LinearRegression_L1,
            X_test = X_test,
            Y_test = Y_test
        ))

    for l2 in tqdm.tqdm(Lambda_2, desc = 'L2'):
        Sklearn_LinearRegression_L2 = models.Sklearn_Ridge(
            Lambda2 = l2,
            MaxIters = configs.MAX_ITERS,
            WithNorm = True,
        )
        Ours_LinearRegression_L2 = models.LinearRegression(
            LearningRate = configs.LEARNING_RATE,
            MaxIters = configs.MAX_ITERS,
            Lambda1 = 0,
            Lambda2 = l2
        )
        Ours_LinearRegression_L2.fit(X_train, Y_train, use_prox_l1 = True)
        Test_Loss_Ours_L2.append(utils.Test(
            Model = Ours_LinearRegression_L2,
            X_test = X_test,
            Y_test = Y_test
        ))
        Sklearn_LinearRegression_L2.fit(X_train, Y_train)
        Test_Loss_Sklearn_L2.append(utils.Test(
            Model = Sklearn_LinearRegression_L2,
            X_test = X_test,
            Y_test = Y_test
        ))

    plt.figure(figsize = (12, 6))
    plt.plot(Lambda_1, Test_Loss_Sklearn_L1, label = 'Sklearn L1', linestyle = '--', linewidth = 4)
    plt.plot(Lambda_1, Test_Loss_Ours_L1, label = 'Ours L1')
    plt.legend()
    plt.xlabel('Lambda')
    plt.ylabel('Test MSE')
    plt.title('Ours vs Sklearn (L1)')
    plt.savefig(os.path.join(configs.SAVE_DIR, 'Q3_comparison_l1.png'))
    plt.close()

    plt.figure(figsize = (12, 6))
    plt.plot(Lambda_2, Test_Loss_Sklearn_L2, label = 'Sklearn L2', linestyle = '--', linewidth = 4)
    plt.plot(Lambda_2, Test_Loss_Ours_L2, label = 'Ours L2')
    plt.legend()
    plt.xlabel('Lambda')
    plt.ylabel('Test MSE')
    plt.title('Ours vs Sklearn (L2)')
    plt.savefig(os.path.join(configs.SAVE_DIR, 'Q3_comparison_l2.png'))
    plt.close()