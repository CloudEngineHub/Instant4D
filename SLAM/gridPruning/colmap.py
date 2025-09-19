import numpy as np




point3d_path= "/data/zhanpeng/nvidia/data/Balloon2/sparse/0/points3D.bin"
camera_path = "/data/zhanpeng/nvidia/data/Balloon2/sparse/0/cameras.bin"

# with open(point3d_path, 'rb') as f:
#     data = f.read()  # Reads the entire binary content into a bytes object

data = np.fromfile(point3d_path, dtype=np.float32)
print(data.shape)

data_camera = np.fromfile(camera_path, dtype=np.float32)
print(data_camera.shape)



