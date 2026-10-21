"""思考题6: 验证集预测结果地图可视化"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import matplotlib.pyplot as plt
import numpy as np
import shapefile

import src.datasets as ds
import src.utils as utils
from configs import Q6_Configs as configs

def draw_land(ax):
    sf = shapefile.Reader(str(Path(configs.MAP_DIR) / 'ne_110m_land' / 'ne_110m_land.shp'))
    for shape in sf.shapes():
        Parts = list(shape.parts) + [len(shape.points)]
        for i in range(len(Parts) - 1):
            Ring = np.array(shape.points[Parts[i]:Parts[i + 1]])
            ax.fill(Ring[:, 0], Ring[:, 1], color = '#e8e8e8', zorder = 1)

if __name__ == "__main__":
    Final, data, Y_pred = ds.load_world_submission()
    Y_true = data['Y_valid']
    Hav = utils.calc_distance(Y_true[:, 0], Y_true[:, 1], Y_pred[:, 0], Y_pred[:, 1])
    print(f"Valid Haversine: median = {np.median(Hav):.1f} km, mean = {np.mean(Hav):.1f} km, <500km: {np.mean(Hav < 500):.1%}")

    plt.figure(figsize = (14, 7))
    ax = plt.gca()
    draw_land(ax)
    ax.plot(
        np.stack([Y_true[:, 1], Y_pred[:, 1]]), 
        np.stack([Y_true[:, 0], Y_pred[:, 0]]),
        '-', color = 'gray', alpha = 0.3, 
        linewidth = 0.5, zorder = 2
    )
    ax.scatter(Y_true[:, 1], Y_true[:, 0], s = 18, c = 'tab:blue', label = 'Ground Truth', zorder = 3)
    ax.scatter(Y_pred[:, 1], Y_pred[:, 0], s = 18, c = 'tab:red', label = 'Predicted', zorder = 4)
    ax.set(xlim = (-180, 180), ylim = (-90, 90), aspect = 1, xlabel = 'Longitude', ylabel = 'Latitude')
    ax.set_title(f"World Probe on Valid ({Final['version']}, layer {Final['layer']}), "
                 f"median error {np.median(Hav):.0f} km")
    ax.legend(loc = 'lower left')
    plt.tight_layout()
    plt.savefig(Path(configs.SAVE_DIR) / 'Q6_world_map.png', dpi = 150)
    plt.close()

    with open(Path(configs.RESULTS_DIR) / 'Q6_valid_errors.json', 'w') as f:
        json.dump(
            {
                'hav_median_km': float(np.median(Hav)),
                'hav_mean_km': float(np.mean(Hav)),
                'frac_below_500km': float(np.mean(Hav < 500)),
                'per_point_km': Hav.tolist()
            },
            f, indent = 2
        )
    print('Figure and Stats Saved.')
