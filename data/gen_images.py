import render
import os
import torch
import torchvision.datasets as dset
import numpy as np
import cv2


def data_loader(file_path):
    with open(file_path, 'rb') as f:
        voxels = render.read_as_3d_array(f)
    fake_voxels = torch.zeros(1, 64, 64, 64)
    chair = np.rot90(voxels.data, 3, (0, 2))
    for a in range(voxels.dims[0]):
        for b in range(voxels.dims[1]):
            for c in range(voxels.dims[2]):
                if chair.data[a, b, c]:
                    fake_voxels[0, a, b, c] = 1
    return fake_voxels


dataset = dset.DatasetFolder(root='.', loader=data_loader, extensions=('.binvox',))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=2)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def main():
    for i, data in enumerate(dataloader):
        voxels = data[0].to(device)
        ots = render.render_tensor(voxels, device)
        cv2.imwrite(f"chair_images/chair_{i}.png", ots.detach().cpu()[0, 0].numpy() * 255)


if __name__ == '__main__':
    # make chair model directory
    if not os.path.isdir('chair_images/'):
        os.mkdir('chair_images/')
    main()
