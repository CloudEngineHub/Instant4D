# read a pickle fild and print the content
import pickle


# path = "/data/zhanpeng/dycheck/apple/dense/colmap_full.pickle"
# with open(path, 'rb') as f:
#     data = pickle.load(f)

# print(type(data))
# # print(data.keys())
# print(data["0_00000.png"])


bin_path = "/data/zhanpeng/dycheck/apple/dense/sparse/points3D.bin"
with open(bin_path, 'rb') as f:
    data = f.read()

print(type(data))
# print(data.keys())
# print(data["points3D"].keys())
