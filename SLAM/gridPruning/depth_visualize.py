import torch
import numpy as np
from PIL import Image
import os


def visualize_img(img, name, time256=False):
    img = img
    if time256:
        img = (img * 255).astype(np.uint8)
    else:
        img = (img ).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(name)


def visualize_droid(npz_path):
    # npz_path = "/data/guest_storage/zhanpengluo/SLAM/msam/mega-sam/outputs/Balloon1_droid.npz"
    data = np.load(npz_path)
    data = {key: data[key] for key in data.keys()}

    Frames = data['images']
    Depths = data['depths']

    B, H, W, _ = Frames.shape
    print(f"H: {H}, W: {W}")

    save_dir = "/data/guest_storage/zhanpengluo/SLAM/msam/mega-sam/visual/depth"
    scene_dir = "/data/guest_storage/zhanpengluo/SLAM/msam/mega-sam/visual/Balloon1"
    os.makedirs(scene_dir, exist_ok=True)
    for i in range(B):
        visualize_img(Depths[i], f"{scene_dir}/depth_{i}.png")


def visualize_da(dir, save_dir):
    for file in os.listdir(dir):
        if file.endswith(".npy"):
            npy_path = os.path.join(dir, file)
            depth = np.load(npy_path)
            file_name = file.split(".")[-2]
            visualize_img(depth, f"{save_dir}/{file_name}.png")

        
visualize_da("/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/medium/Depth-Anything/Balloon2"
             ,"/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/medium/visualization/Balloon2DepthAnything")



