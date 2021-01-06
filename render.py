import pyrender
import numpy as np
import trimesh
import cv2
import torch


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


def mesh_colors():
    r = [1.0, 0.0, 0.0]
    g = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 1.0]
    return [r, g, r, b, b, g, b, g, b, g, r, r]


sm = trimesh.creation.box((1, 1, 1))
sm.visual.face_colors = mesh_colors()

camera = pyrender.OrthographicCamera(xmag=45, ymag=45, zfar=1000)
camera_pose = np.array([
    [-6.43333333e-01, +3.83333333e-01, -6.66666666e-01, -2.33333333e+01],
    [+0.00000000e+00, +8.66666666e-01, +5.00000000e-01, +7.25000000e+01],
    [+7.66666666e-01, +3.33333333e-01, -5.66666666e-01, -1.50000000e+01],
    [+0.00000000e+00, +0.00000000e+00, +0.00000000e+00, +1.00000000e+00]
])
scene = pyrender.Scene(bg_color=np.ones(4), ambient_light=np.ones(3))
camera_node = scene.add(camera, pose=camera_pose)


def render_canonical(fake_voxels, is_grayscale, threshold=0.5):
    points = (fake_voxels > threshold).nonzero(as_tuple=False).cpu()

    tfs = np.tile(np.eye(4), (len(points), 1, 1))
    tfs[:, :3, 3] = points
    mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs, smooth=False)
    mesh_node = scene.add(mesh)

    r = pyrender.OffscreenRenderer(viewport_width=256, viewport_height=256, point_size=1)
    color, depth = r.render(scene)
    r.delete()
    scene.remove_node(mesh_node)

    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if is_grayscale else color


def render_tensor(fake_voxels, device):
    ots_results = []
    for i in range(fake_voxels.shape[0]):
        img = np.expand_dims(render_canonical(fake_voxels[i, 0], True), axis=(0, 1))
        ots_results.append(img)
    return torch.tensor(np.vstack(ots_results), dtype=torch.float).to(device) / 255
