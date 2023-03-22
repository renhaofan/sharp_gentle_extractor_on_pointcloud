import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import open3d as o3d
import numpy as np
import torch
from GDANet_util import GDM
import time
from os.path import join


def farthest_point_down_sample(vertices, num_point_sampled):
    # 最远点采样 FPS # vertices.shape = (N,3) or (N,2)
    N = len(vertices)
    n = num_point_sampled
    assert n <= N, "Num of sampled point should be less than or equal to the size of vertices."
    _G = np.mean(vertices, axis=0) # centroid of vertices
    _d = np.linalg.norm(vertices - _G, axis=1, ord=2)
    farthest = np.argmax(_d) # 取离重心最远的点为起始点
    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_) # 点是否被选中
    for i in range(n):
        flags[farthest] = True
        distances[farthest] = 0.
        p_farthest = vertices[farthest]
        dists = np.linalg.norm(vertices[~flags] - p_farthest, axis=1, ord=2)
        distances[~flags] = np.minimum(distances[~flags], dists)
        farthest = np.argmax(distances)
    return vertices[flags]


ply_file = "assets/bed_0614.ply"
# pcd_raw = o3d.io.read_point_cloud("bed_0614.ply")
# pcd_raw = o3d.io.read_point_cloud("bunny10k.ply")
pcd_raw = o3d.io.read_point_cloud(ply_file)
# pcd_raw = o3d.io.read_point_cloud("lego_ngp_official.ply")
print(f'raw shape: {pcd_raw}')


# pcd = pcd_raw.uniform_down_sample(100)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(farthest_point_down_sample(np.asarray(pcd_raw.points), 1024))

pcd.paint_uniform_color([1, 0.706, 0])
# pcd = o3d.io.read_point_cloud("bunny10k.ply")
# pcd.paint_uniform_color([0.5, 0.5, 0.5])
# o3d.visualization.draw_geometries([pcd])

# convert Open3D.o3d.geometry.PointCloud to numpy array
xyz_load = np.asarray(pcd.points)

xyz = torch.tensor(xyz_load)
xyz = xyz.unsqueeze(0)
xyz = xyz.permute(0, 2, 1)
print(f'shape extracted: {xyz.shape}')

# samples_n = int(xyz.shape[2] * 0.25)
samples_n = 256
print(f'Sample: {samples_n}')


#sharp, gentle
start = time.time()
x1s, x1g = GDM(xyz, M=samples_n)
# print(x1s.shape, x1g.shape)
print(f'Elapsted time: {(time.time() - start):.2f} s')

x1s = x1s.squeeze(0)
x1g = x1g.squeeze(0)
print(x1s.shape, x1g.shape)

sharp_o3d = o3d.geometry.PointCloud()
sharp_o3d.points = o3d.utility.Vector3dVector(x1s)
sharp_o3d.paint_uniform_color([1, 0, 0])

gentle_o3d = o3d.geometry.PointCloud()
gentle_o3d.points = o3d.utility.Vector3dVector(x1g)
gentle_o3d.paint_uniform_color([0, 0.5, 1])

pcd.translate((3, 0, 0))
o3d.visualization.draw_geometries([pcd, gentle_o3d, sharp_o3d])

ret_path = 'ret'
if not os.path.exists(ret_path):
    os.mkdir(ret_path)
timeprefix = time.strftime("%Y%m%d-%H%M%S")
o3d.io.write_point_cloud(join(ret_path, timeprefix + "-sharp-" + ply_file.split('/')[-1]), sharp_o3d)
o3d.io.write_point_cloud(join(ret_path, timeprefix + "-gentle-" + ply_file.split('/')[-1]), gentle_o3d)


