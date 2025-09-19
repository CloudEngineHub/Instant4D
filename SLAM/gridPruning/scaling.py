import numpy as np
import open3d as o3d
import cv2
import point_cloud_utils as pcu
import json
from typing import NamedTuple
import os
import glob 
class pcd (NamedTuple):
    xyz: np.ndarray
    rgb: np.ndarray
    prob_motion: np.ndarray
    time_stamp: np.ndarray

def back_project(depth, intrinsic, cam_c2w):
    """
    Vectorized back-projection of depth maps to 3D points in world coordinates.
    
    Args:
        depth: B, H, W numpy array
        intrinsic: 3, 3 numpy array
        cam_c2w: B, 4, 4 numpy array
    
    Returns:
        xyz: B, H*W, 3 numpy array of 3D points in world coordinates
    """
    B, H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1) + 0.5  # Add 0.5 for pixel center
    y = y.reshape(-1) + 0.5
    
    # Create homogeneous coordinates
    homogeneous_coords = np.vstack((x, y, np.ones_like(x)))
    
    # Apply inverse intrinsics
    cam_points = np.linalg.inv(intrinsic) @ homogeneous_coords  # 3 x (H*W)
    
    # Reshape depth and multiply
    depth_flat = depth.reshape(B, -1)  # B x (H*W)
    
    # Scale points by depth for each batch
    # Expand cam_points to B x 3 x (H*W)
    cam_points_expanded = np.tile(cam_points[None, :, :], (B, 1, 1))
    
    # Multiply by depth along the correct dimension
    cam_points_scaled = cam_points_expanded * depth_flat[:, None, :]  # B x 3 x (H*W)
    
    # Transform to world coordinates
    world_points = np.zeros((B, H*W, 3))
    for b in range(B):
        world_points[b] = (cam_points_scaled[b].T @ cam_c2w[b, :3, :3].T) + cam_c2w[b, :3, 3]
    
    return world_points

def read_dycheck_data(droid_path, motion_path, save_dir):
    droid_data = np.load(droid_path)
    color = droid_data['images'] # B, H, W, 3
    depth = droid_data['depths'] # B, H, W
    B, H, W = depth.shape
    intrinsic = droid_data['intrinsic'] # 3, 3
    cam_c2w = droid_data['cam_c2w'] # B, 4, 4

    mask_png_list = sorted(glob.glob(f"{motion_path}/*.png"))
    motion_bool_list = []
    assert len(mask_png_list) != 0
    for mask_path in mask_png_list:
        base_name = os.path.basename(mask_path)
        group_key = base_name.split('_')[0]
        if group_key != "0":
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))
        mask = mask!=255
        mask = mask.reshape(-1, 1)
        motion_bool_list.append(mask)

    color       = color[2:-2]
    depth       = depth[2:-2]
    cam_c2w     = cam_c2w[2:-2]    
    
    

    

    return depth, color, motion_bool_list, intrinsic, cam_c2w

def process_data(depth, color, motion_prob, intrinsic, cam_c2w):
    B, H, W = depth.shape

    xyz = back_project(depth, intrinsic, cam_c2w).reshape(-1, 3)
    rgb = color.reshape(-1, 3).astype(np.float32)/255.0
    
    time_stamp = np.repeat(np.arange(B).astype(np.float32)/B*3,
                           xyz.shape[0]//B)
    time_stamp = time_stamp.reshape(-1, 1)
    
    prob_motion = motion_prob
    # print(f"prob_motion range from {np.min(prob_motion)} to {np.max(prob_motion)}")
    
    pc = pcd(xyz=xyz, rgb=rgb, prob_motion=prob_motion, time_stamp=time_stamp)
    
    return pc

def dynamic_static_split(pc, threshold=0.7):
    dynamic_region = pc.prob_motion[:,0] > threshold
    static_region = ~dynamic_region
    

    xyz_dynamic = pc.xyz[dynamic_region]
    rgb_dynamic = pc.rgb[dynamic_region]
    prob_motion_dynamic = pc.prob_motion[dynamic_region]
    time_stamp_dynamic = pc.time_stamp[dynamic_region]

    xyz_static = pc.xyz[static_region]
    rgb_static = pc.rgb[static_region]
    prob_motion_static = pc.prob_motion[static_region]
    time_stamp_static  = pc.time_stamp[static_region]
    
    dynamic_pcd = pcd(xyz=xyz_dynamic, rgb=rgb_dynamic, prob_motion=prob_motion_dynamic, time_stamp=time_stamp_dynamic)
    static_pcd =  pcd(xyz=xyz_static,  rgb=rgb_static,  prob_motion=prob_motion_static,  time_stamp=time_stamp_static)
    
    return dynamic_pcd, static_pcd

def make_transforms(intrinsic, cam_c2w, save_dir, scene ,W=480):
    scale_factor = 360 / W
    B = cam_c2w.shape[0]
    print(f"cam_c2w: {cam_c2w.shape}")

    dict_to_save = {}
    dict_to_save["w"]    = 480
    dict_to_save["h"]    = 270
    dict_to_save["fl_x"] = (intrinsic[0, 0] * scale_factor).item()
    dict_to_save["fl_y"] = (intrinsic[1, 1] * scale_factor).item()
    dict_to_save["cx"]   = (intrinsic[0, 2] * scale_factor).item()
    dict_to_save["cy"]   = (intrinsic[1, 2] * scale_factor).item()
    frame = []

    dycheck_path = "/data/zhanpeng/dycheck"
    
    # we split the frame to 85% train and 15% test
    m = int(B * 0.85)
    print(f"------------ split train: {m} test: {B-m}")
    
    selected_indices = np.linspace(0, B-1, num=m, dtype=np.int32)  # evenly spaced indices
    selected =   [i for i in selected_indices]
    remaining =  [i for i in range(B) if i not in selected_indices]

    print(f"selected_len: {len(selected)}")
    print(f"remaining_len: {len(remaining)}")
    
    train_frame = []
    for i in selected:
        frame_dict = {
            "file_path": f"{dycheck_path}/{scene}/dense/rgb/2x/0_{i:05d}",
            "transform_matrix": cam_c2w[i].tolist(),
            "time": i/(B-1)*3
        }
        train_frame.append(frame_dict)
    
    dict_to_save["frames"] = train_frame

    with open(f"{save_dir}/transforms_train.json", "w") as f:
        json.dump(dict_to_save, f, indent=4)
        
        
    test_frame = []
    for i in remaining:
        frame_dict = {
            "file_path": f"{dycheck_path}/{scene}/dense/rgb/2x/0_{i:05d}",
            "transform_matrix": cam_c2w[i].tolist(),
            "time": i/(B-1)*3
        }
        test_frame.append(frame_dict)
    
    dict_to_save["frames"] = test_frame
    
    with open(f"{save_dir}/transforms_test.json", "w") as f:
        json.dump(dict_to_save, f, indent=4)
    


def filter_once(color, depth, intrinsic, cam_c2w, voxel_size, motion_prob):
    pc = process_data(depth, color, motion_prob, intrinsic, cam_c2w)
    pcd_dynamic, pcd_static = dynamic_static_split(pc)
    
    mean_depth = np.mean(depth[0])
    focal = intrinsic[0, 0]

    
    xyz_static= pcu.downsample_point_cloud_on_voxel_grid(voxel_size*1.5,pcd_static.xyz)
    xyz_dynamic= pcu.downsample_point_cloud_on_voxel_grid(voxel_size*0.7,pcd_dynamic.xyz)
    
    return xyz_static.shape[0], xyz_dynamic.shape[0]


def voxel_filter(droid_path, motion_path, save_dir, scene, use_mask=False):    
    if not use_mask:
        depth, color, motion_prob, intrinsic, cam_c2w = read_droid_data(droid_path, motion_path, save_dir)
    else:
        depth, color, motion_prob, intrinsic, cam_c2w = read_dycheck_data(droid_path, motion_path, save_dir)
        
    B, H, W = depth.shape
    
    # we want to keep track of the number of points as the frame number increases
    num_points_static = []
    num_points_dynamic = []
    
    # motion_prob = np.concatenate(motion_prob, axis=0)
    # turn motion prob from list to numpy array
    motion_prob = np.array(motion_prob)
    
    motion_prob = motion_prob.astype(np.float32)
    

    color_base = color[0][None]
    depth_base = depth[0][None]
    motion_prob_base = motion_prob[0]
    intrinsic_base = intrinsic
    cam_c2w_base = cam_c2w[0][None]
    # 
    # motion_prob_base: (196608, 1)
    # color_base: (1, 512, 384, 3)
    # depth_base: (1, 512, 384)
    # intrinsic_base: (3, 3)
    # cam_c2w_base: (1, 4, 4)
    
    
    mean_depth = np.mean(depth[0])
    focal = intrinsic[0, 0]

    voxel_size_dynamic = mean_depth / focal * 4
    voxel_size_static = mean_depth / focal * 4
    
    for i in range(2,B,10):
        color_this = np.concatenate([color_base, color[1:i]], axis=0)
        depth_this = np.concatenate([depth_base, depth[1:i]], axis=0)
        motion_total = np.concatenate(motion_prob[1:i], axis=0)
        motion_prob_this = np.concatenate([motion_prob_base, motion_total], axis=0)
        intrinsic_this = intrinsic
        cam_c2w_this = np.concatenate([cam_c2w_base, cam_c2w[1:i]], axis=0)
        # print(f"motion_prob_this: {motion_prob_this.shape}")
        # print(f"color_this: {color_this.shape}")
        # print(f"depth_this: {depth_this.shape}")
        # print(f"intrinsic_this: {intrinsic_this.shape}")
        # print(f"cam_c2w_this: {cam_c2w_this.shape}")
        # raise Exception("stop here")
        
        points_static, points_dynamic = filter_once(color_this, depth_this, intrinsic_this, cam_c2w_this, voxel_size_static, motion_prob_this)
        print(f"For frame {i}, num_points_static: {points_static}, num_points_dynamic: {points_dynamic}")


        

    

if __name__ == "__main__":
    
    use_msam = False
    scene_list = [ "block", "spin", "teddy", "apple", "paper-windmill"]
    scene_list = [ "paper-windmill"]
    droid_dir = "/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/outputs_cvd"
    motion_dir_msam = "/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/reconstructions"
    save_dir = "/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/voxel_filter/output/densfication_COLMAP"
    
    motion_dir_mask_dir =  "/data/zhanpeng/dycheck"
    
    for scene in scene_list:
        if use_msam:
            droid_path = f"{droid_dir}/{scene}_sgd_cvd_hr.npz"
            motion_path = f"{motion_dir_msam}/{scene}/motion_prob.npy"
            save_path = f"{save_dir}/{scene}"
            os.makedirs(save_path, exist_ok=True)
            voxel_filter(droid_path, motion_path, save_path, scene, use_mask=False)
        else:
            droid_path = f"{droid_dir}/{scene}_sgd_cvd_hr.npz"
            motion_path = f"{motion_dir_mask_dir}/{scene}/dense/masks/"
            save_path = f"{save_dir}/{scene}"
            os.makedirs(save_path, exist_ok=True)
            voxel_filter(droid_path, motion_path, save_path, scene, use_mask=True)

