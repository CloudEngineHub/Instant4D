

# print the value range of the mask
# path /data/zhanpeng/dycheck/teddy/dense/masks

import os
import cv2
import numpy as np
import glob

mask_dir = "/data/zhanpeng/dycheck/block/dense/masks"
mask_files = glob.glob(os.path.join(mask_dir, "*.png"))

mask_file = mask_files[1]

mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
print(mask.shape)
print(mask.min(), mask.max())

# print the unique values
print(np.unique(mask))


# visualize the mask == 2 part 
mask_2 = mask == 3

# save the mask_2
cv2.imwrite("mask_2.png", mask_2.astype(np.uint8)*255)


# teddy 2 3 4 255
# paper-windmill 0 255
# block 0 1 255
# apple 0 1 255
# spin 0 255