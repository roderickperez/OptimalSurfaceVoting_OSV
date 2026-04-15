import numpy as np
import torch
from torch.utils.data import Dataset
import os


class DataGenerator(Dataset):
    def __init__(self, dpath, fpath, data_IDs, dim=(128,128,128),
                 split="train", augment=False, vertical_axis_post=0):
        self.dpath = dpath
        self.fpath = fpath
        self.data_IDs = data_IDs
        self.dim = dim

        # NEW:
        self.split = split              # "train" | "val" | "test"
        self.augment = augment          # enable only for training
        # Because you call np.transpose(...) with no axes (reverses to (2,1,0)),
        # vertical=z ends up at axis 0 after that step → default to 0 here.
        self.vertical_axis_post = vertical_axis_post

    def __len__(self):
        base = len(self.data_IDs)
        if self.split == "train" and self.augment:
            return base * 2
        return base

    def __getitem__(self, index):
        if self.split == "train" and self.augment:
            original_index = index // 2
        else:
            original_index = index
            
        ID = self.data_IDs[original_index]

        # 1) Load + Reshape
        gx = np.load(os.path.join(self.dpath, f"{ID}.npy")).astype(np.single)
        fx = np.load(os.path.join(self.fpath, f"{ID}.npy")).astype(np.single)
        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)

        # 2) Per-cube z-score
        xm = gx.mean(); xs = gx.std()
        gx = (gx - xm) / xs

        # 3) Transpose (matches your existing logic: reverses axes to (2,1,0))
        #    So vertical=z becomes axis 0
        gx = np.transpose(gx)
        fx = np.transpose(fx)

        # 4) Make labels binary
        fx = (fx > 0).astype(np.single)

        # 5) Augmentation & Batching
        if self.split == "train" and self.augment:
            # We process files with indices up to 2 * len(data_IDs)
            # if index is odd, we flip it.
            vertical_axis = self.vertical_axis_post   # 0
            horiz_axes = tuple(ax for ax in range(3) if ax != vertical_axis)
            
            # Random rotation k in {0,1,2,3}
            # (Note: doing this per-item might mean original/flip pairs 
            # get different rotations, which is fine for augmentation)
            k = np.random.randint(0, 4)
            if k:
                gx = np.rot90(gx, k=k, axes=horiz_axes).copy()
                fx = np.rot90(fx, k=k, axes=horiz_axes).copy()

            # Flip if it's the "augmented" half of the dataset
            # (e.g. index >= len(data_IDs), or alternating)
            # A common way:
            flip_item = (index % 2 == 1)
            if flip_item:
                gx = np.flip(gx, axis=vertical_axis).copy()
                fx = np.flip(fx, axis=vertical_axis).copy()

            # Shape -> (1, D, H, W)
            gx = gx[None, ...] # (1, D, H, W)
            fx = fx[None, ...] # (1, D, H, W)

            X = torch.from_numpy(np.ascontiguousarray(gx)).float()
            Y = torch.from_numpy(np.ascontiguousarray(fx)).float()

        else:
            # Validation / No augmentation -> Return single item batch
            # Shape (1, D, H, W)
            gx = gx[None, ...] # (1, D, H, W)
            fx = fx[None, ...] # (1, D, H, W)

            X = torch.from_numpy(np.ascontiguousarray(gx)).float()
            Y = torch.from_numpy(np.ascontiguousarray(fx)).float()

        return X, Y
