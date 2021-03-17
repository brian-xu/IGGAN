import numpy as np
import torch
import argparse

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


def gen_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--nr_lr', type=float, default=2e-5)
    parser.add_argument('--gan_lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--dom_lambda', type=float, default=100)
    parser.add_argument('--z_size', type=float, default=200)
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--dropout_rate', type=float, default=0.25)
    parser.add_argument('--is_grayscale', type=bool, default=True)
    parser.add_argument('--models', type=int, default=5)
    return parser
