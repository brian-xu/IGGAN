import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import model

# Set random seed for reproducibility
import render

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

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

args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

if args.is_grayscale:
    img_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])
else:
    img_transform = transforms.ToTensor()

dataset = dset.ImageFolder(root=args.dataset,
                           transform=img_transform)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers)


def generator_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def discriminator_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight_bar.data, 0.0, 0.02)
        nn.init.normal_(m.weight_u.data, 0.0, 0.02)
        nn.init.normal_(m.weight_v.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def renderer_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)


generator = model.Generator(args).to(device)
generator.apply(generator_init)

discriminator = model.Discriminator(args).to(device)
discriminator.apply(discriminator_init)

renderer = model.RenderNet(args).to(device)
renderer.apply(renderer_init)


def DOMLoss(x, y):
    return torch.mean(torch.square(torch.log(x) - torch.log(y)))


criterion = nn.BCELoss()

l2 = nn.MSELoss(reduction='sum')

# fixed_noise = torch.randn(64, args.z_size, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optimizerG = optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(args.beta1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(args.beta1, 0.999))
optimizerR = optim.Adam(renderer.parameters(), lr=args.nr_lr, betas=(args.beta1, 0.999))


# G_losses = []
# D_losses = []
# R_losses = []


def main():
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.z_size, 1, 1, device=device)
            # Generate fake voxels batch with G
            fake_voxels = generator(noise)
            # Render fake voxel batch with R
            fake = renderer(fake_voxels)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            ############################
            # (3) Update R network: minimize l2(R) + lambda * DOM(R)
            ###########################
            renderer.zero_grad()
            # Since we just updated G, generate a new fake batch
            fake_voxels = generator(noise)
            # Render the voxels with the neural renderer
            nr = renderer(fake_voxels)
            # Render the voxels with an off-the-shelf renderer
            ots_results = []
            for ex in range(fake_voxels.shape[0]):
                img = np.expand_dims(render.render_canonical(fake_voxels[ex, 0], True), axis=(0, 1))
                ots_results.append(img)
            ots = torch.tensor(np.vstack(ots_results), dtype=torch.float).to(device) / 255
            # Perform a forward pass of neural renderer output through D
            nr_output = discriminator(nr).view(-1)
            # Perform a forward pass of off-the-shelf renderer output through D
            ots_output = discriminator(ots).view(-1)
            # Calculate R's L2 loss based on squared error of pixel matrix
            errL2 = l2(nr, ots)
            # Calculate R's DOM loss based on squared log error of discriminator output
            errDOM = DOMLoss(ots_output, nr_output)
            errR = errL2 + args.dom_lambda * errDOM
            # Calculate gradients for R
            errR.backward()
            R_x = nr_output.mean().item()
            # Update R
            optimizerR.step()

            # Output training stats
            if i % 50 == 0:
                print(f"[{epoch}/{args.num_epochs}][{i}/{len(dataloader)}]\t"
                      f"Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tLoss_R: {errR.item():.4f}\t"
                      f"D(x):  {D_x:.4f}\tD(G(z)):  {D_G_z1:.4f}/ {D_G_z2:.4f}\tR(G(z):  {R_x:.4f}")

            # # Save Losses for plotting later
            # G_losses.append(errG.item())
            # D_losses.append(errD.item())
            # R_losses.append(errR.item())

            # with torch.no_grad():
            #     fake = generator(fixed_noise).detach().cpu()


if __name__ == '__main__':
    main()
