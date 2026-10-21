"""思考题4: 对比不同层中间表征的预测效果(base vs extra;与 Q5 同口径,原始特征空间多输出 Ridge,λ2 固定)"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import src.datasets as ds
import src.models as models
import src.utils as utils
from configs import Q4_Configs as configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-exist', action = argparse.BooleanOptionalAction, default = True)
    parser.add_argument('--layers', nargs = '+', type = int, default = None,
                        help = f'默认全 {configs.N_LAYERS} 层')
    args = parser.parse_args()
    Layers = args.layers if args.layers else list(range(configs.N_LAYERS))

    Results = {}
    for layer in Layers:
        Results[layer] = {}
        for version in ('base', 'extra'):
            data = ds.load_world(version, layer = layer, use_exist = args.use_exist)
            m = models.Sklearn_Ridge(Lambda2 = configs.LAMBDA2)
            m.fit(data['X_train'], data['Y_train'])
            Results[layer][version] = utils.eval_world(data['Y_valid'], m.predict(data['X_valid']))
            print(f"layer {layer} [{version}]: {Results[layer][version]}")

    with open(Path(configs.RESULTS_DIR) / 'Q4_layer_sweep.json', 'w') as f:
        json.dump({str(l): Results[l] for l in Layers}, f, indent = 2)

    Layers_Arr = np.array(Layers)
    fig, Axes = plt.subplots(1, 2, figsize = (12, 5))
    for version, label in (('base', 'Base'), ('extra', 'Extra')):
        Hav = [Results[l][version]['hav_median_km'] for l in Layers]
        Mse = [Results[l][version]['mse_lat'] + Results[l][version]['mse_lon'] for l in Layers]
        Axes[0].plot(Layers_Arr, Hav, '-o', markersize = 3, label = label)
        Axes[1].plot(Layers_Arr, Mse, '-o', markersize = 3, label = label)
    Axes[0].set(title = "World Probe: Layer Sweep", xlabel = "Layer", ylabel = "Valid Median Haversine Error (km)")
    Axes[1].set(title = "World Probe: Layer Sweep (MSE)", xlabel = "Layer", ylabel = "Valid MSE (Lat + Lon)")
    for ax in Axes:
        ax.legend()
        ax.grid(alpha = 0.3)
    plt.tight_layout()
    plt.savefig(Path(configs.SAVE_DIR) / "Q4_layer_sweep.png")
    plt.close()
    print("Figure Saved.")
