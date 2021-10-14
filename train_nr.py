import render
import model
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import utils
import os

parser = utils.gen_parser()

args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

dataset = dset.DatasetFolder(root=args.dataset, loader=utils.data_loader, extensions=('.binvox',))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers)


def renderer_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)


renderer = model.RenderNet(args).to(device)
if os.path.exists("weights/nr.pt"):
    renderer.load_state_dict(torch.load('weights/nr.pt'))
else:
    renderer.apply(renderer_init)

criterion = nn.BCELoss()

l2 = nn.MSELoss(reduction='sum')

optimizerR = optim.Adam(renderer.parameters(), lr=args.nr_lr, betas=(args.beta1, 0.999))

img_list = []


def main():
    for epoch in range(args.num_epochs):
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
                print(f"[{epoch}/{args.num_epochs}][{i}/{len(dataloader)}]\t Loss_R: {errR.item():.4f}\t")
            if i % 200 == 0:
                img_list.append(nr.detach().cpu()[0, 0].numpy() * 255)


if __name__ == '__main__':
    main()
    torch.save(renderer.state_dict(), 'weights/nr.pt')
    import matplotlib; matplotlib.use("TkAgg")  # uncomment if using PyCharm
    fig, ax = plt.subplots()
    plt.axis("off")
    ims = [[plt.imshow(i, cmap='gray', animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)
    plt.show()
