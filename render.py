import pyrender
import numpy as np
import trimesh
import cv2
import torch


def mesh_colors():
    r = [1.0, 0.0, 0.0]
    g = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 1.0]
    return [r, g, r, b, b, g, b, g, b, g, r, r]


sm = trimesh.creation.box((1, 1, 1))
sm.visual.face_colors = mesh_colors()

camera = pyrender.OrthographicCamera(xmag=50, ymag=50, zfar=1000)
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
