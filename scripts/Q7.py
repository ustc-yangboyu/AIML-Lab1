"""思考题7: 地理坐标表征 vs 国家识别(国家均值基线对比)"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import matplotlib.pyplot as plt
import numpy as np

import src.datasets as ds
import src.utils as utils
from configs import Q7_Configs as configs

if __name__ == "__main__":
    Final, data, Y_pred = ds.load_world_submission()
    Y_true = data['Y_valid']

    Country_Means = {
        c: data['Y_train'][(data['Train_Countries'] == c).to_numpy()].mean(axis = 0)
        for c in data['Train_Countries'].unique()
    }
    Global_Mean = data['Y_train'].mean(axis = 0)
    Y_base = np.stack([Country_Means.get(c, Global_Mean) for c in data['Valid_Countries']])

    Hav_Probe = utils.calc_distance(Y_true[:, 0], Y_true[:, 1], Y_pred[:, 0], Y_pred[:, 1])
    Hav_Base = utils.calc_distance(Y_true[:, 0], Y_true[:, 1], Y_base[:, 0], Y_base[:, 1])
    print(f"Overall | probe hav_med = {np.median(Hav_Probe):.1f} km, country-mean baseline hav_med = {np.median(Hav_Base):.1f} km")

    Masks = {c: (data['Valid_Countries'] == c).to_numpy() for c in data['Valid_Countries'].unique()}
    Per_Country = [
        (c, int(m.sum()), float(np.median(Hav_Probe[m])), float(np.median(Hav_Base[m])))
        for c, m in Masks.items() if m.sum() >= configs.MIN_VALID_PER_COUNTRY and c in Country_Means
    ]
    Per_Country.sort(key = lambda t: t[3] - t[2], reverse = True)
    Countries = [t[0] for t in Per_Country]

    X_Pos = np.arange(len(Per_Country))
    plt.figure(figsize = (14, 6))
    plt.bar(X_Pos - 0.2, [t[3] for t in Per_Country], width = 0.4, label = 'Country-Mean Baseline', color = 'tab:orange')
    plt.bar(X_Pos + 0.2, [t[2] for t in Per_Country], width = 0.4, label = 'Linear Probe', color = 'tab:blue')
    for i, (_, n, p, b) in enumerate(Per_Country):
        plt.text(i, max(p, b) + 20, str(n), ha = 'center', fontsize = 7)
    plt.xticks(X_Pos, Countries, rotation = 60, ha = 'right', fontsize = 8)
    plt.ylabel('Median Haversine Error (km)')
    plt.title('Within-Country Errors on Valid')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(configs.SAVE_DIR) / 'Q7_country_baseline.png')
    plt.close()

    Show_Countries = ['United States', 'China', 'Australia', 'India']
    plt.figure(figsize = (14, 4))
    for i, country in enumerate(Show_Countries):
        m = Masks[country]
        plt.subplot(1, 4, i + 1)
        plt.scatter(Y_true[m, 1], Y_true[m, 0], s = 25, c = 'tab:green', label = 'Truth')
        plt.scatter(Y_pred[m, 1], Y_pred[m, 0], s = 25, c = 'tab:red', marker = 'x', label = 'Probe Pred')
        plt.scatter(Y_base[m, 1], Y_base[m, 0], s = 60, c = 'tab:orange', marker = '*', label = 'Baseline')
        plt.title(country, fontsize = 10)
        plt.xlabel('Longitude')
        if i == 0:
            plt.ylabel('Latitude')
    plt.legend(fontsize = 7, loc = 'best')
    plt.tight_layout()
    plt.savefig(Path(configs.SAVE_DIR) / 'Q7_within_country_scatter.png')
    plt.close()

    Correlations = {}
    for c, m in Masks.items():
        if m.sum() < 4 or c not in Country_Means:
            continue
        Corr_Lat = np.corrcoef(Y_true[m, 0], Y_pred[m, 0])[0, 1] if np.var(Y_true[m, 0]) > 0 else float('nan')
        Corr_Lon = np.corrcoef(Y_true[m, 1], Y_pred[m, 1])[0, 1] if np.var(Y_true[m, 1]) > 0 else float('nan')
        Correlations[c] = {
            'n': int(m.sum()), 
            'corr_lat': float(Corr_Lat), 
            'corr_lon': float(Corr_Lon),
            'probe_mean_km': float(np.mean(Hav_Probe[m])),
            'baseline_mean_km': float(np.mean(Hav_Base[m]))
        }

    Wins = sum(1 for _, _, p, b in Per_Country if p < b)
    with open(Path(configs.RESULTS_DIR) / 'Q7_country_comparison.json', 'w') as f:
        json.dump(
            {
                'overall_probe_hav_median_km': float(np.median(Hav_Probe)),
                'overall_baseline_hav_median_km': float(np.median(Hav_Base)),
                'probe_wins': Wins, 
                'n_countries': len(Per_Country),
                'within_country_correlations': Correlations,
                'per_country': {
                    c: {
                        'probe_med_km': p, 
                        'baseline_med_km': b, 
                        'n_valid': n
                    }
                    for c, n, p, b in Per_Country
                }
            }, 
            f, indent = 2
        )
    print('Figures and Stats Saved.')
