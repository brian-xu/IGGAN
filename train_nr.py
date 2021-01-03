import numpy as np
import render
import model
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import cv2


class Voxels(object):
    """
    Holds a binvox model.
    data is either a three-dimensional numpy boolean array (dense representation)
    or a two-dimensional numpy float array (coordinate representation).
    dims, translate and scale are the model metadata.
    dims are the voxel dimensions, e.g. [32, 32, 32] for a 32x32x32 model.
    scale and translate relate the voxels to the original model coordinates.
    To translate voxel coordinates i, j, k to original coordinates x, y, z:
    x_n = (i+.5)/dims[0]
    y_n = (j+.5)/dims[1]
    z_n = (k+.5)/dims[2]
    x = scale*x_n + translate[0]
    y = scale*y_n + translate[1]
    z = scale*z_n + translate[2]
    """

    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)


def read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale


def read_as_3d_array(fp, fix_coords=True):
    """ Read binary binvox format as array.
    Returns the model with accompanying metadata.
    Voxels are stored in a three-dimensional numpy array, which is simple and
    direct, but may use a lot of memory for large models. (Storage requirements
    are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
    boolean arrays use a byte per element).
    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    return Voxels(data, dims, translate, scale, axis_order)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="data/")
parser.add_argument('--bias', type=bool, default=True)
parser.add_argument('--dropout_rate', type=float, default=0.25)
parser.add_argument('--is_grayscale', type=bool, default=True)

args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

batch_size = 1
num_epochs = 10
workers = 2
nr_lr = 2e-5
beta1 = 0.5


def data_loader(file_path):
    with open(file_path, 'rb') as f:
        voxels = read_as_3d_array(f)
    fake_voxels = torch.zeros(1, 64, 64, 64)
    for a in range(voxels.dims[0]):
        for b in range(voxels.dims[1]):
            for c in range(voxels.dims[2]):
                if voxels.data[a, b, c]:
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


def main():
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            voxels = data[0].to(device)
            ############################
            # (1) Update R network: minimize l2(R)
            ###########################
            renderer.zero_grad()
            # Render the voxels with the neural renderer
            nr = renderer(voxels)
            # Render the voxels with an off-the-shelf renderer
            ots_results = []
            for ex in range(voxels.shape[0]):
                img = np.expand_dims(render.render_canonical(voxels[ex, 0], True), axis=(0, 1))
                ots_results.append(img)
            ots = torch.tensor(np.vstack(ots_results), dtype=torch.float).to(device) / 255
            # Calculate R's L2 loss based on squared error of pixel matrix
            errR = l2(nr, ots)
            # Calculate gradients for R
            errR.backward()
            # Update R
            optimizerR.step()
            # Output training stats
            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t Loss_R: {errR.item():.4f}\t")
            if i % 100 == 0:
                cv2.imwrite(f'testing/nr_{epoch}_{i}.jpg', nr.detach().cpu()[0, 0].numpy() * 255)


if __name__ == '__main__':
    main()
    torch.save(renderer.state_dict(), 'nr.pt')
