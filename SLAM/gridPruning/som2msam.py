import numpy as np
import os
import glob

depth_dir = "/data/zhanpeng/som-dycheck/apple/flow3d_preprocessed/aligned_depth_anything_colmap/1x"
# 0_00000.npy  0_00001.npy ....

depth_list = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))

for depth_path in depth_list:
    depth = np.load(depth_path)
    print(depth.shape)
    print(depth_path.split("/")[-1])
