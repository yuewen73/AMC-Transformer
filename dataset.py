# dataset.py
import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def dataset_split(
    data, modulations_classes, modulations, snrs,
    target_modulations, target_snrs,
    mode, train_proportion=0.7, valid_proportion=0.2, test_proportion=0.1,
    seed=42
):
    np.random.seed(seed)

    X_out, Y_out, Z_out = [], [], []
    target_idx = [modulations_classes.index(m) for m in target_modulations]

    for m in target_idx:
        for snr in target_snrs:
            idx = np.where((modulations == m) & (snrs == snr))[0]
            if len(idx) == 0:
                continue
            np.random.shuffle(idx)
            N = min(len(idx), 4096)
            idx = idx[:N]

            n_tr = int(train_proportion * N)
            n_va = int((train_proportion + valid_proportion) * N)

            if mode == "train":
                sel = idx[:n_tr]
            elif mode == "valid":
                sel = idx[n_tr:n_va]
            else:
                sel = idx[n_va:]

            X_out.append(data[sel])
            Y_out.append(modulations[sel])
            Z_out.append(snrs[sel])

    X = np.vstack(X_out)
    Y = np.concatenate(Y_out)
    Z = np.concatenate(Z_out)

    for i, v in enumerate(np.unique(Y)):
        Y[Y == v] = i

    return X, Y, Z


class RadioML18Dataset(Dataset):
    def __init__(self, data_dir, mode="train", seed=42):
        h5_path = os.path.join(data_dir, "GOLD_XYZ_OSC.0001_1024.hdf5")
        cls_path = os.path.join(data_dir, "classes-fixed.json")
        
        with h5py.File(h5_path, "r") as h5:
            X = h5["X"][:]
            Y = np.argmax(h5["Y"][:], axis=1)
            Z = h5["Z"][:, 0]
        
        with open(cls_path, "r") as f:
            classes = json.load(f)
        

        self.target_modulations = classes
        self.target_snrs = np.unique(Z)

        self.X, self.Y, self.Z = dataset_split(
            X, classes, Y, Z,
            target_modulations=self.target_modulations,
            target_snrs=self.target_snrs,
            mode=mode,
            seed=seed
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx]).transpose(0, 1)
        y = self.Y[idx]
        z = self.Z[idx]
        return x, y, z
