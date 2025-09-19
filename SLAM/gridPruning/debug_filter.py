import numpy as np

# Load the two .npz files
file1 = np.load('/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/voxel_filter/output/test_1/paper-windmill/filtered_cvd.npz')
file2 = np.load('/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/voxel_filter/output/test_0/paper-windmill/filtered_cvd.npz')

# Check what keys (arrays) each file has
keys1 = set(file1.keys())
keys2 = set(file2.keys())

# Find keys that are different
keys_only_in_file1 = keys1 - keys2
keys_only_in_file2 = keys2 - keys1
common_keys = keys1 & keys2

print("Keys only in file1:", keys_only_in_file1)
print("Keys only in file2:", keys_only_in_file2)
print("Common keys:", common_keys)

# Now compare arrays for common keys
for key in common_keys:
    array1 = file1[key]
    array2 = file2[key]
    if not np.array_equal(array1, array2):
        print(f"Difference found in key: {key}")
    else:
        print(f"Key {key} is identical.")
