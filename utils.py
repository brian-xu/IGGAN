import numpy as np
import torch

import binvox_rw


def data_loader(file_path):
    with open(file_path, 'rb') as f:
        voxels = binvox_rw.read_as_3d_array(f)
    fake_voxels = torch.zeros(1, 64, 64, 64)
    model = np.rot90(voxels.data, 3, (0, 2))
    for a in range(voxels.dims[0]):
        for b in range(voxels.dims[1]):
            for c in range(voxels.dims[2]):
                if model.data[a, b, c]:
                    fake_voxels[0, a, b, c] = 1
    return fake_voxels
