import numpy as np
import render
import model
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import matplotlib.animation as animation

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="data/")
parser.add_argument('--bias', type=bool, default=True)
parser.add_argument('--dropout_rate', type=float, default=0.25)
parser.add_argument('--is_grayscale', type=bool, default=True)

args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

batch_size = 1
num_epochs = 5
workers = 2
nr_lr = 2e-5
beta1 = 0.5


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


dataset = dset.DatasetFolder(root=args.dataset, loader=data_loader, extensions=('.binvox',))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


def renderer_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)


renderer = model.RenderNet(args).to(device)
renderer.apply(renderer_init)

criterion = nn.BCELoss()

l2 = nn.MSELoss(reduction='sum')

optimizerR = optim.Adam(renderer.parameters(), lr=nr_lr, betas=(beta1, 0.999))

img_list = []


def main():
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            ############################
            # (1) Update R network: minimize l2(R)
            ###########################
            renderer.zero_grad()
            voxels = data[0].to(device)
            # Render the voxels with the neural renderer
            nr = renderer(voxels)
            # Render the voxels with an off-the-shelf renderer
            ots = render.render_tensor(voxels, device)
            # Calculate R's L2 loss based on squared error of pixel matrix
            errR = l2(nr, ots)
            # Calculate gradients for R
            errR.backward()
            # Update R
            optimizerR.step()
            del ots

            # Output training stats
            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t Loss_R: {errR.item():.4f}\t")
            if i % 200 == 0:
                img_list.append(nr.detach().cpu()[0, 0].numpy() * 255)
            print(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())


if __name__ == '__main__':
    main()
    torch.save(renderer.state_dict(), 'nr.pt')
    # import matplotlib; matplotlib.use("TkAgg")  # uncomment if using PyCharm
    fig, ax = plt.subplots()
    plt.axis("off")
    ims = [[plt.imshow(i, cmap='gray', animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)
    plt.show()
