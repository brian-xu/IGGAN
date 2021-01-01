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

camera = pyrender.OrthographicCamera(xmag=52, ymag=52, zfar=1000)
camera_pose = np.array([
    [-6.43043726e-01, 3.82967760e-01, -6.63165959e-01, -2.27456165e+01],
    [2.54106097e-04, 8.65931585e-01, 4.99881402e-01, 7.32862870e+01],
    [7.65963251e-01, 3.21222953e-01, -5.56908875e-01, -1.39929237e+01],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
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
