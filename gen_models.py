import argparse
import torch
import model
import os
import numpy as np
import binvox_rw
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--z_size', type=float, default=200)
parser.add_argument('--bias', type=bool, default=True)
parser.add_argument('--models', type=int, default=5)

args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

generator = model.Generator(args).to(device)
generator.load_state_dict(torch.load('weights/gen.pt'))
generator.eval()

first_model = next(os.scandir('data/chair_models/'))
with open(first_model, 'rb') as f:
    ref_vox = binvox_rw.read_as_3d_array(f)

timestamp = datetime.datetime.today().strftime('%Y%m%d%H%M%S')


def main():
    with torch.no_grad():
        for i in range(args.models):
            noise = torch.randn(1, args.z_size, 1, 1, device=device)
            # Generate fake voxels batch with G
            fake_voxels = generator(noise)
            voxels = np.array(fake_voxels.detach().cpu()[0, 0, :, :, :].numpy(), dtype=bool)
            result = binvox_rw.Voxels(voxels, ref_vox.dims, ref_vox.translate, ref_vox.scale, ref_vox.axis_order)
            with open(f'{timestamp}_{i}.binvox', 'wb') as f:
                binvox_rw.write(result, f)


if __name__ == '__main__':
    main()
