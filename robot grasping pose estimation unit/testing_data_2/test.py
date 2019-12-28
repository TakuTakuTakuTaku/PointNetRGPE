import numpy as np
from open3d import *


pcd = read_point_cloud("47.pcd")
draw_geometries([pcd])

