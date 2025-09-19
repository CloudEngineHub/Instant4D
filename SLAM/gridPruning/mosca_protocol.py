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
    print(droid_data.keys())
    # images, depths, intrinsic, cam_c2w, motion_prob
    print(droid_data['images'].shape)
    print(droid_data['depths'].shape)
    print(droid_data['intrinsic'].shape)
    print(droid_data['cam_c2w'].shape)

    color = droid_data['images'] # B, H, W, 3
    depth = droid_data['depths'] # B, H, W
    B, H, W = depth.shape
    intrinsic = droid_data['intrinsic'] # 3, 3
    cam_c2w = droid_data['cam_c2w'] # B, 4, 4

    mask_png_list = sorted(glob.glob(f"{motion_path}/*.png"))
    motion_bool_list = []
    print(len(mask_png_list))
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
        
    test1_cam_c2w = cam_c2w[0]
    test2_cam_c2w = cam_c2w[1]

    color       = color[2:-2]
    depth       = depth[2:-2]
    cam_c2w     = cam_c2w[2:-2]    
    
    
    print(color.shape)
    print(depth.shape)
    print(len(motion_bool_list))
    

    return depth, color, motion_bool_list, intrinsic, cam_c2w, test1_cam_c2w, test2_cam_c2w

def process_data(depth, color, motion_prob, intrinsic, cam_c2w):
    B, H, W = depth.shape

    xyz = back_project(depth, intrinsic, cam_c2w).reshape(-1, 3)
    rgb = color.reshape(-1, 3).astype(np.float32)/255.0
    
    time_stamp = np.repeat(np.arange(B).astype(np.float32)/B*3,
                           xyz.shape[0]//B)
    time_stamp = time_stamp.reshape(-1, 1)
    
    prob_motion = motion_prob
    print(f"prob_motion range from {np.min(prob_motion)} to {np.max(prob_motion)}")
    
    pc = pcd(xyz=xyz, rgb=rgb, prob_motion=prob_motion, time_stamp=time_stamp)
    
    return pc

def dynamic_static_split(pc, threshold=0.7):
    dynamic_region = pc.prob_motion[:,0] > threshold
    static_region = ~dynamic_region
    
    print(f"shape of dynamic region: {dynamic_region.shape}")
    print(f"shape of static region: {static_region.shape}")
    print(f"shape of pc xyz: {pc.xyz.shape}")

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

def make_transforms(intrinsic, cam_c2w, save_dir, scene, test1_cam_c2w, test2_cam_c2w ,W=480):
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
    
    # we read all the pictures named 0_{i:05d}.png as the train set 
    # all the pictures named 1_{i:05d}.png and 2_{i:05d}.png as the test set

    dycheck_path = "/data/zhanpeng/dycheck"
    train_list = sorted(glob.glob(f"{dycheck_path}/{scene}/dense/rgb/2x/0_*.png"))
    test_list = sorted(glob.glob(f"{dycheck_path}/{scene}/dense/rgb/2x/1_*.png"))
    test_list = test_list + sorted(glob.glob(f"{dycheck_path}/{scene}/dense/rgb/2x/2_*.png"))
    
    # we read all the pictures named 0_{i:05d}.png as the train set 
    # all the pictures named 1_{i:05d}.png and 2_{i:05d}.png as the test set
    
    train_frame = []
    for i, train_path in enumerate(train_list):
        if f"{dycheck_path}/{scene}/dense/rgb/2x/0_{i:05d}.png" != train_path:
            print(f"--------------------------------")
            print(f"{dycheck_path}/{scene}/dense/rgb/2x/0_{i:05d}.png")
            print(train_path)
            raise Exception(f"train_path {train_path} is not correct")
        # get rid of the .png
        frame_dict = {
            "file_path": train_path.split(".")[0],
            "transform_matrix": cam_c2w[i].tolist(),
            "time": i/(B-1)*3
        }
        train_frame.append(frame_dict)
    dict_to_save["frames"] = train_frame

    with open(f"{save_dir}/transforms_train.json", "w") as f:
        json.dump(dict_to_save, f, indent=4)
        
    test_frame = []
    for i, test_path in enumerate(test_list):
        # get the base name of the test path, get the group and the index, get rid of the .png
        base_name = os.path.basename(test_path)
        group = base_name.split("_")[0]
        index = base_name.split("_")[1]
        index = int(index.split(".")[0])
        if group == "1":
            test_cam_c2w = test1_cam_c2w
        if group == "2":
            test_cam_c2w = test2_cam_c2w
        
        frame_dict = {
            "file_path": test_path.split(".")[0],
            "transform_matrix": test_cam_c2w.tolist(),
            "time": index/(B-1)*3
        }
        
        test_frame.append(frame_dict)
    
    dict_to_save["frames"] = test_frame
    
    with open(f"{save_dir}/transforms_test.json", "w") as f:
        json.dump(dict_to_save, f, indent=4)
        
    
def voxel_filter(droid_path, motion_path, save_dir, scene, use_mask=False):
    if not use_mask:
        depth, color, motion_prob, intrinsic, cam_c2w = read_droid_data(droid_path, motion_path, save_dir)
    else:
        depth, color, motion_prob, intrinsic, cam_c2w, test1_cam_c2w, test2_cam_c2w = read_dycheck_data(droid_path, motion_path, save_dir)
        
    B, H, W = depth.shape
    print(f"depth shape: {depth.shape}")
    make_transforms(intrinsic, cam_c2w, save_dir, scene, test1_cam_c2w, test2_cam_c2w, W)

    
    # select every 10th frame 
    # n H W 
    color       = color[::10]
    depth       = depth[::10]
    cam_c2w     = cam_c2w[::10]
    motion_prob = motion_prob[::10]
    
    motion_prob = np.concatenate(motion_prob, axis=0)
    motion_prob = motion_prob.astype(np.float32)
    
    print(f"motion_prob shape: {motion_prob.shape}")
    print(f"color shape: {color.shape}")
    print(f"depth shape: {depth.shape}")
    print(f"cam_c2w shape: {cam_c2w.shape}")
    
    
    pc = process_data(depth, color, motion_prob, intrinsic, cam_c2w)
    pcd_dynamic, pcd_static = dynamic_static_split(pc)
    
    mean_depth = np.mean(depth[0])
    focal = intrinsic[0, 0]

    # voxel size for dynamic region
    voxel_size_dynamic = mean_depth / focal * 4
    voxel_size_static = mean_depth / focal * 4
    
    DoSample = False
    if scene == "paper-windmill" or scene == "spin":
        voxel_size_static = mean_depth / focal * 4
        voxel_size_dynamic = mean_depth / focal * 6
        DoSample = False
    else:
        DoSample = True
    
    if scene == "teddy":
        voxel_size_static = mean_depth / focal * 4
        voxel_size_dynamic = mean_depth / focal * 4
    if scene == "apple":
        voxel_size_static = mean_depth / focal * 6
        voxel_size_dynamic = mean_depth / focal * 6
    if scene == "spin":
        voxel_size_static = mean_depth / focal * 6
        voxel_size_dynamic = mean_depth / focal * 6
    
    xyz_static, rgb_static, prob_motion_static= pcu.downsample_point_cloud_on_voxel_grid(voxel_size_static,
                                                                                         pcd_static.xyz,
                                                                                         pcd_static.rgb,
                                                                                         pcd_static.prob_motion)
    if DoSample:        
        xyz_dynamic, rgb_dynamic, prob_motion_dynamic, time_stamp_dynamic= pcu.downsample_point_cloud_on_voxel_grid(voxel_size_dynamic,
                                                                                            pcd_dynamic.xyz,
                                                                                            pcd_dynamic.rgb,
                                                                                            pcd_dynamic.prob_motion,
                                                                                            pcd_dynamic.time_stamp)
    else:
        xyz_dynamic = pcd_dynamic.xyz
        rgb_dynamic = pcd_dynamic.rgb
        prob_motion_dynamic = pcd_dynamic.prob_motion
        time_stamp_dynamic = pcd_dynamic.time_stamp
    
    time_stamp_static = np.repeat(1, xyz_static.shape[0])
    scale_time_static = np.repeat(3, xyz_static.shape[0])
    scale_time_dynamic = np.repeat(3/((B-1)*10), time_stamp_dynamic.shape[0])

    xyz_sampled = np.concatenate([xyz_static, xyz_dynamic], axis=0)
    rgb_sampled = np.concatenate([rgb_static, rgb_dynamic], axis=0)
    prob_motion_sampled = np.concatenate([prob_motion_static, prob_motion_dynamic.squeeze()], axis=0)
    time_stamp_sampled =  np.concatenate([time_stamp_static,  time_stamp_dynamic.squeeze()], axis=0)
    scale_time_sampled = np.concatenate([scale_time_static, scale_time_dynamic], axis=0)
    
    print(f"--------------------------------")
    print(f"Scene: {scene}")
    print(f"xyz_static: {xyz_static.shape}")
    print(f"xyz_dynamic: {xyz_dynamic.shape}")
    print(f"xyz_sampled: {xyz_sampled.shape}")
    print(f"time_stamp: {time_stamp_sampled.shape}")
    print(f"prob_motion: {prob_motion_sampled.shape}")
    print(f"scale_time: {scale_time_sampled.shape}")
    np.savez(f"{save_dir}/filtered_cvd.npz", 
            xyz=xyz_sampled,
            rgb=rgb_sampled,
            prob_motion=prob_motion_sampled,
            time_stamp=time_stamp_sampled,
            scale_time=scale_time_sampled,
            intrinsic=intrinsic,
            cam_c2w=cam_c2w)

if __name__ == "__main__":
    
    use_msam = False
    scene_list = [ "block", "spin", "teddy", "apple", "paper-windmill"]
    # scene_list = [ "teddy"]
    droid_dir = "/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/outputs_cvd"
    motion_dir_msam = "/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/reconstructions"
    save_dir = "/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/voxel_filter/output/mosca"
    
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

