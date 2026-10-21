"""思考题2"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import src.models as models
import src.datasets as ds
import src.utils as utils
import numpy as np
import tqdm
import os

from configs import Q2_Configs as configs

np.random.seed(configs.SEED)

class LinearRegression_Clip(models.LinearRegression):
    def weights_clip(self, eps):
        self.Weights[np.abs(self.Weights) < eps] = 0.0

if __name__ == "__main__":
    data = ds.diabetes()
    X, Y, Name = data['X'], data['Y'], data['Name']
    Extra_X = X[:, configs.COPYED_FEATURES]
    Copy_Count = {}
    Extra_Name = []
    for feat in configs.COPYED_FEATURES:
        Copy_Count[feat] = Copy_Count.get(feat, 0) + 1
        Extra_Name.append(Name[feat] + "_" + str(Copy_Count[feat]))
    Name = Name + Extra_Name
    Extra_X = Extra_X + np.random.normal(0, configs.NOISE_STD, Extra_X.shape)
    X = np.concat((X, Extra_X), axis = 1)
    X_train, Y_train, X_test, Y_test = ds.split_datasets(X, Y, test_size = configs.TEST_RATIO).values()

    Linear_model_l1 = LinearRegression_Clip(
        LearningRate = configs.LEARNING_RATE,
        MaxIters = configs.MAX_ITERS,
        Lambda1 = configs.LAMBDA1,
        Lambda2 = 0.0
    )

    Linear_model_l2 = LinearRegression_Clip(
        LearningRate = configs.LEARNING_RATE,
        MaxIters = configs.MAX_ITERS,
        Lambda1 = 0.0,
        Lambda2 = configs.LAMBDA2
    )

    print("Training Started...")

    Linear_model_l1.fit(X_train, Y_train)
    Linear_model_l2.fit(X_train, Y_train)

    print("Finished.")

    l1_weights_origin = Linear_model_l1.Weights
    l2_weights_origin = Linear_model_l2.Weights

    print(f"L1 Weights: {l1_weights_origin}\n")
    print(f"L2 Weights: {l2_weights_origin}\n")

    L1_Loss = utils.Test(Linear_model_l1, X_test, Y_test)
    L2_Loss = utils.Test(Linear_model_l2, X_test, Y_test)
    print(f"L1 Test Loss: {L1_Loss:.4f}")
    print(f"L2 Test Loss: {L2_Loss:.4f}")

    Linear_model_l1.Weights = l1_weights_origin.copy()
    Linear_model_l2.Weights = l2_weights_origin.copy()
    Linear_model_l1.weights_clip(eps = configs.CLIP_EPS_SHOW_L1)
    Linear_model_l2.weights_clip(eps = configs.CLIP_EPS_SHOW_L2)
    L1to0_Loss = utils.Test(Linear_model_l1, X_test, Y_test)
    L2to0_Loss = utils.Test(Linear_model_l2, X_test, Y_test)
    l1to0_weights_show = Linear_model_l1.Weights
    l2to0_weights_show = Linear_model_l2.Weights
    print(f"L1to0 (Clip Eps: {configs.CLIP_EPS_SHOW_L1}) Test Loss: {L1to0_Loss:.4f}")
    print(f"L2to0 (Clip Eps: {configs.CLIP_EPS_SHOW_L2}) Test Loss: {L2to0_Loss:.4f}")

    plt.figure(figsize = (12, 6))
    plt.plot(Name, l1_weights_origin, '-o', label = f"L1 (Loss: {L1_Loss:.4f})")
    plt.plot(Name, l2_weights_origin, '-o', label = f"L2 (Loss: {L2_Loss:.4f})")
    plt.plot(Name, l1to0_weights_show, '-o', label = f"L1to0 (Eps: {configs.CLIP_EPS_SHOW_L1}, Loss: {L1to0_Loss:.4f})")
    plt.plot(Name, l2to0_weights_show, '-o', label = f"L2to0 (Eps: {configs.CLIP_EPS_SHOW_L2}, Loss: {L2to0_Loss:.4f})")
    plt.plot([0, len(Name) - 1], [0, 0], 'r--', label = "Zero Line")
    plt.xlabel("Features")
    plt.ylabel("Weights")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(configs.SAVE_DIR, "Q2_weights_comparison.png"))
    plt.close()

    Max_Weight = max(np.max(np.abs(l1_weights_origin)), np.max(np.abs(l2_weights_origin)))
    Clip_Eps = np.linspace(0, Max_Weight, configs.N_CLIP_EPS)

    L1to0_Losses, L2to0_Losses = [], []
    for eps in Clip_Eps:
        Linear_model_l1.Weights = l1_weights_origin.copy()
        Linear_model_l1.weights_clip(eps = eps)
        L1to0_Losses.append(utils.Test(Linear_model_l1, X_test, Y_test))
        Linear_model_l2.Weights = l2_weights_origin.copy()
        Linear_model_l2.weights_clip(eps = eps)
        L2to0_Losses.append(utils.Test(Linear_model_l2, X_test, Y_test))
        print(f"Clip Eps: {eps:.4f}, L1to0 Loss: {L1to0_Losses[-1]:.4f}, L2to0 Loss: {L2to0_Losses[-1]:.4f}")

    plt.figure()
    plt.plot(Clip_Eps, L1to0_Losses, '-o', markersize = 3, label = f"L1to0 (Loss: {L1_Loss:.4f})")
    plt.plot(Clip_Eps, L2to0_Losses, '-o', markersize = 3, label = f"L2to0 (Loss: {L2_Loss:.4f})")
    plt.xlabel("Clip Eps")
    plt.ylabel("Test Loss")
    plt.title("Test Loss / Clip Eps")
    plt.legend()
    plt.savefig(os.path.join(configs.SAVE_DIR, "Q2_clip_test_loss.png"))
    plt.close()
