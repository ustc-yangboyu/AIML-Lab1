from pathlib import Path

from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import csv

def Test(Model, X_test, Y_test):
    Y_pred = Model.predict(X_test)
    return Model.LossFunc(Y_pred, Y_test)

def pick_token(row, offsets, selection):
    if selection == "last":
        return [j for j, (l, r) in enumerate(offsets) if r > l][-1]
    start = row["sentence"].rfind(row["country"])
    end = start + len(row["country"])
    return [j for j, (l, r) in enumerate(offsets) if r > l and l < end and r > start][-1]

def extract_layers(input_csv, output_dir, layers, token_selection = "country"):
    from configs import Extract_Configs as configs

    rows = list(csv.DictReader(Path(input_csv).open(encoding = "utf-8-sig")))
    tokenizer = AutoTokenizer.from_pretrained(configs.MODEL_DIR, local_files_only = True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        configs.MODEL_DIR, local_files_only = True,
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ).to("cuda").eval()
    model.norm = torch.nn.Identity()

    out = {L: np.empty((len(rows), model.config.hidden_size), np.float16) for L in layers}
    with torch.inference_mode():
        for k in range(0, len(rows), configs.BATCH_SIZE):
            batch = rows[k:k + configs.BATCH_SIZE]
            enc = tokenizer(
                [r["sentence"] for r in batch], padding = True,
                return_tensors = "pt", return_offsets_mapping = True,
            )
            spans = enc.pop("offset_mapping").tolist()
            index = [pick_token(r, s, token_selection) for r, s in zip(batch, spans)]
            hs = model(**enc.to("cuda"), use_cache = False, output_hidden_states = True).hidden_states
            sel = torch.arange(len(index), device = "cuda")
            for L in layers:
                out[L][k:k + len(batch)] = hs[L + 1][sel, index, :].float().cpu().numpy()
            print(f"Extracted {k + len(batch)}/{len(rows)}", flush = True)

    Path(output_dir).mkdir(parents = True, exist_ok = True)
    for L, feats in out.items():
        np.save(Path(output_dir) / f"{Path(input_csv).stem}.layer{L}.npy", feats)
    print(f"Saved {len(out)} layers to {output_dir}")

def zscore(X, Mean = None, Std = None):
    if Mean is None:
        Mean = np.mean(X, axis = 0)
    if Std is None:
        Std = np.std(X, axis = 0)
    return (X - Mean) / Std, Mean, Std

def calc_distance(Lat1, Lon1, Lat2, Lon2):
    P1, P2 = np.radians(Lat1), np.radians(Lat2)
    DPhi = P2 - P1
    DLon = np.radians(Lon2 - Lon1)
    A = np.sin(DPhi / 2) ** 2 + np.cos(P1) * np.cos(P2) * np.sin(DLon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(A))

def eval_world(Y_true, Y_pred):
    Hav = calc_distance(Y_true[:, 0], Y_true[:, 1], Y_pred[:, 0], Y_pred[:, 1])
    return {'mse_lat': float(np.mean((Y_pred[:, 0] - Y_true[:, 0]) ** 2)),
            'mse_lon': float(np.mean((Y_pred[:, 1] - Y_true[:, 1]) ** 2)),
            'hav_mean_km': float(np.mean(Hav)), 'hav_median_km': float(np.median(Hav))}

def eval_acc(Score, Y):
    return float(np.mean((Score > 0) == (Y == 1)))
