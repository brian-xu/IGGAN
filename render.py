import pyrender
import numpy as np
import trimesh
import cv2


def mesh_colors():
    r = [1.0, 0.0, 0.0]
    g = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 1.0]
    return [r, g, r, b, b, g, b, g, b, g, r, r]


sm = trimesh.creation.box((1, 1, 1))
sm.visual.face_colors = mesh_colors()

camera = pyrender.OrthographicCamera(xmag=45, ymag=45, zfar=1000)
camera_pose = np.array([
    [-4.87500000e-01, -2.50000000e-01, +8.33333333e-01, +1.00000000e+02],
    [-0.00000000e+00, +9.66666666e-01, +2.93333333e-01, +5.55872594e+01],
    [-8.73333333e-01, +1.43333333e-01, -4.66666666e-01, -6.62382094e+00],
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
