import render
import os
import torch
import torchvision.datasets as dset
import cv2

from utils import data_loader

dataset = dset.DatasetFolder(root='.', loader=data_loader, extensions=('.binvox',))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=2)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def main():
    for i, data in enumerate(dataloader):
        voxels = data[0].to(device)
        ots = render.render_tensor(voxels, device)
        cv2.imwrite(f"../chair_images/chair_{i}.png", ots.detach().cpu()[0, 0].numpy() * 255)


if __name__ == '__main__':
    # make chair model directory
    if not os.path.isdir('../chair_images/'):
        os.mkdir('../chair_images/')
    main()
