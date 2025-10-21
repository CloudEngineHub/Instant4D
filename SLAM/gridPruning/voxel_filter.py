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

def read_droid_data(droid_path, motion_path, save_dir):
    droid_data = np.load(droid_path)
    print(droid_data.keys())
    # images, depths, intrinsic, cam_c2w, motion_prob
    print(droid_data['images'].shape)
    print(droid_data['depths'].shape)
    print(droid_data['intrinsic'].shape)
    print(droid_data['cam_c2w'].shape)

    color = droid_data['images'] # B, H, W, 3
    depth = droid_data['depths'] # B, H, W
    intrinsic = droid_data['intrinsic'] # 3, 3
    cam_c2w = droid_data['cam_c2w'] # B, 4, 4
    motion_prob = np.load(motion_path)

    print(motion_prob.shape)

    # color = np.concatenate([color[12:13], color[1:11], color[13:14]], axis=0)
    # depth = np.concatenate([depth[12:13], depth[1:11], depth[13:14]], axis=0)
    # cam_c2w = np.concatenate([cam_c2w[12:13], cam_c2w[1:11], cam_c2w[13:14]], axis=0)
    # motion_prob = np.concatenate([motion_prob[12:13], motion_prob[1:11], motion_prob[13:14]], axis=0)
    
    # # save each of color into a png file
    # for i in range(color.shape[0]):
    #     color_resized = cv2.resize(color[i], (480, 270))
    #     color_resized = color_resized[:, :, ::-1]
    #     cv2.imwrite(f"{save_dir}/img/color_{i}.png", color_resized)

    return depth, color, motion_prob, intrinsic, cam_c2w

def read_droid_data_mask(droid_path, motion_path, save_dir):
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
    for mask_path in mask_png_list:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))
        mask = mask>0
        mask = mask.reshape(-1, 1)
        motion_bool_list.append(mask)
    motion_bool = np.concatenate(motion_bool_list, axis=0)
    motion_prob = motion_bool.astype(np.float32)
    
    print(motion_prob.shape)
    print(depth.shape)

    
    return depth, color, motion_prob, intrinsic, cam_c2w

def process_data(depth, color, motion_prob, intrinsic, cam_c2w):
    B, H, W = depth.shape

    xyz = back_project(depth, intrinsic, cam_c2w).reshape(-1, 3)
    rgb = color.reshape(-1, 3).astype(np.float32)/255.0
    
    # we want to make a tensor of t for each frame's point cloud
    time_stamp = np.repeat(np.arange(B).astype(np.float32)/B, xyz.shape[0]//B)
    time_stamp = time_stamp.reshape(-1, 1)
    # print(f"dtype time_stamp: {time_stamp.dtype}")

    # motion_prob_list = []
    # for i in range(B):
    #     motion_prob_list.append(cv2.resize(motion_prob[i], (W, H)))
    # prob_motion = np.stack(motion_prob_list, axis=0).reshape(-1,1) 


    
    prob_motion = motion_prob
    print(f"prob_motion range from {np.min(prob_motion)} to {np.max(prob_motion)}")
    
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
    scale_factor = 480 / W
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

    nvidia_path = "/data/zhanpeng/nvidia/data"
    test_path = "/home/zhanpeng/code/SLAM/reconstruct/Instant4dGS/fourth_party/robust-dynrf/results/Nvidia/gt"
    for i in range(B):
        frame_dict = {
            "file_path": f"{nvidia_path}/{scene}/images_2/{i:03d}",
            "transform_matrix": cam_c2w[i].tolist(),
            "time": i/(B-1)
        }
        frame.append(frame_dict)
    
    dict_to_save["frames"] = frame

    with open(f"{save_dir}/transforms_train.json", "w") as f:
        json.dump(dict_to_save, f, indent=4)
        
        
    test_frame = []
    for i in range(B):
        frame_dict = {
            "file_path": f"{test_path}/{scene}/v000_t{i:03d}",
            "transform_matrix": cam_c2w[0].tolist(),
            "time": i/(B-1)
        }
        test_frame.append(frame_dict)
    
    dict_to_save["frames"] = test_frame
    
    with open(f"{save_dir}/transforms_test.json", "w") as f:
        json.dump(dict_to_save, f, indent=4)

def voxel_filter(droid_path, motion_path, save_dir, scene, use_mask=False):
    if not use_mask:
        depth, color, motion_prob, intrinsic, cam_c2w = read_droid_data(droid_path, motion_path, save_dir)
    else:
        depth, color, motion_prob, intrinsic, cam_c2w = read_droid_data_mask(droid_path, motion_path, save_dir)
    
    B, H, W = depth.shape
    pc = process_data(depth, color, motion_prob, intrinsic, cam_c2w)
    pcd_dynamic, pcd_static = dynamic_static_split(pc)
    
    mean_depth = np.mean(depth[0])
    focal = intrinsic[0, 0]

    # voxel size for dynamic region
    voxel_size_dynamic = mean_depth / focal * 3
    voxel_size_static = mean_depth / focal * 4
    
    xyz_static, rgb_static, prob_motion_static= pcu.downsample_point_cloud_on_voxel_grid(voxel_size_static,
                                                                                         pcd_static.xyz,
                                                                                         pcd_static.rgb,
                                                                                         pcd_static.prob_motion)
    
    

    time_stamp_static = np.repeat(1/2, xyz_static.shape[0])
    scale_time_static = np.repeat(1, xyz_static.shape[0])
    scale_time_dynamic = np.repeat(1/((B-1)*10), pcd_dynamic.xyz.shape[0])
    

    xyz_sampled = np.concatenate([xyz_static, pcd_dynamic.xyz], axis=0)
    rgb_sampled = np.concatenate([rgb_static, pcd_dynamic.rgb], axis=0)
    prob_motion_sampled = np.concatenate([prob_motion_static, pcd_dynamic.prob_motion.squeeze()], axis=0)
    time_stamp_sampled =  np.concatenate([time_stamp_static,  pcd_dynamic.time_stamp.squeeze()], axis=0)
    scale_time_sampled = np.concatenate([scale_time_static, scale_time_dynamic], axis=0)

    print(f"--------------------------------")
    print(f"Scene: {scene}")
    print(f"xyz_static: {xyz_static.shape}")
    print(f"xyz_dynamic: {pcd_dynamic.xyz.shape}")
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
    
    make_transforms(intrinsic, cam_c2w, save_dir, scene, W)


def do_not_filter(droid_path, motion_path, save_dir, scene, use_mask=False):
    if not use_mask:
        depth, color, motion_prob, intrinsic, cam_c2w = read_droid_data(droid_path, motion_path, save_dir)
    else:
        depth, color, motion_prob, intrinsic, cam_c2w = read_droid_data_mask(droid_path, motion_path, save_dir)
    
    B, H, W = depth.shape
    pc = process_data(depth, color, motion_prob, intrinsic, cam_c2w)
    pcd_dynamic, pcd_static = dynamic_static_split(pc)
    
    mean_depth = np.mean(depth[0])
    focal = intrinsic[0, 0]


    
    xyz_static, rgb_static, prob_motion_static= pcd_static.xyz, pcd_static.rgb, pcd_static.prob_motion
    
    

    time_stamp_static = np.repeat(1/2, xyz_static.shape[0])
    scale_time_static = np.repeat(1, xyz_static.shape[0])
    scale_time_dynamic = np.repeat(1/((B-1)*10), pcd_dynamic.xyz.shape[0])
    

    xyz_sampled = np.concatenate([xyz_static, pcd_dynamic.xyz], axis=0)
    rgb_sampled = np.concatenate([rgb_static, pcd_dynamic.rgb], axis=0)
    prob_motion_sampled = np.concatenate([prob_motion_static.squeeze(), pcd_dynamic.prob_motion.squeeze()], axis=0)
    time_stamp_sampled =  np.concatenate([time_stamp_static.squeeze(),  pcd_dynamic.time_stamp.squeeze()], axis=0)
    scale_time_sampled = np.concatenate([scale_time_static.squeeze(), scale_time_dynamic.squeeze()], axis=0)
    # xyz_sampled = xyz_static
    # rgb_sampled = rgb_static
    # prob_motion_sampled = prob_motion_static
    # time_stamp_sampled = time_stamp_static
    # scale_time_sampled = scale_time_static
    
    print(f"--------------------------------")
    print(f"Scene: {scene}")
    print(f"xyz_static: {xyz_static.shape}")
    print(f"xyz_dynamic: {pcd_dynamic.xyz.shape}")
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
    
    make_transforms(intrinsic, cam_c2w, save_dir, scene, W)

def voxel_filter_colmap(droid_path, motion_path, save_dir, scene, use_mask=False):
    if not use_mask:
        depth, color, motion_prob, intrinsic, cam_c2w = read_droid_data(droid_path, motion_path, save_dir)
    else:
        depth, color, motion_prob, intrinsic, cam_c2w = read_droid_data_mask(droid_path, motion_path, save_dir)
    
    B, H, W = depth.shape
    pc = process_data(depth, color, motion_prob, intrinsic, cam_c2w)
    pcd_dynamic, pcd_static = dynamic_static_split(pc)
    
    mean_depth = np.mean(depth[0])
    focal = intrinsic[0, 0]

    # voxel size for dynamic region
    voxel_size_static = mean_depth / focal * 5
    
    xyz_static, rgb_static, prob_motion_static= pcu.downsample_point_cloud_on_voxel_grid(voxel_size_static,
                                                                                         pcd_static.xyz,
                                                                                         pcd_static.rgb,
                                                                                         pcd_static.prob_motion)
    

    
    time_stamp_static = np.repeat(1/2, xyz_static.shape[0])
    scale_time_static = np.repeat(1, xyz_static.shape[0])
    scale_time_dynamic = np.repeat(1/((B-1)*10), pcd_dynamic.xyz.shape[0])

    # xyz_sampled = np.concatenate([xyz_static, pcd_dynamic.xyz], axis=0)
    # rgb_sampled = np.concatenate([rgb_static, pcd_dynamic.rgb], axis=0)
    # prob_motion_sampled = np.concatenate([prob_motion_static.squeeze(), pcd_dynamic.prob_motion.squeeze()], axis=0)
    # time_stamp_sampled =  np.concatenate([time_stamp_static.squeeze(),  pcd_dynamic.time_stamp.squeeze()], axis=0)
    # scale_time_sampled = np.concatenate([scale_time_static.squeeze(), scale_time_dynamic.squeeze()], axis=0)
    
    xyz_sampled = xyz_static
    rgb_sampled = rgb_static
    prob_motion_sampled = prob_motion_static.squeeze()
    time_stamp_sampled = time_stamp_static.squeeze()
    scale_time_sampled = scale_time_static.squeeze()

    print(f"--------------------------------")
    print(f"Scene: {scene}")
    print(f"xyz_static: {xyz_static.shape}")
    print(f"xyz_dynamic: {pcd_dynamic.xyz.shape}")
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
    make_transforms(intrinsic, cam_c2w, save_dir, scene, W)



if __name__ == "__main__":
    
    use_msam = False

    scene_list = ["Balloon1", "Balloon2", "Jumping", "Playground", "Skating", "Truck", "Umbrella"]
    scene_list = ["Umbrella"]
    
    droid_dir = "/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/outputs_cvd"
    motion_dir_msam = "/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/reconstructions"
    save_dir = "/home/zhanpeng/code/SLAM/reconstruct/MegaSAM_GS/voxel_filter/output/nvidia"
    
    motion_dir_mask_dir =  "/data/zhanpeng/nvidia/data"
    
    for scene in scene_list:
        if use_msam:
            droid_path = f"{droid_dir}/{scene}_sgd_cvd_hr.npz"
            motion_path = f"{motion_dir_msam}/{scene}/motion_prob.npy"
            save_path = f"{save_dir}/{scene}"
            os.makedirs(save_path, exist_ok=True)
            voxel_filter(droid_path, motion_path, save_path, scene, use_mask=False)
        else:
            droid_path = f"{droid_dir}/{scene}_sgd_cvd_hr.npz"
            motion_path = f"{motion_dir_mask_dir}/{scene}/motion_masks/"
            save_path = f"{save_dir}/{scene}"
            os.makedirs(save_path, exist_ok=True)
            voxel_filter(droid_path, motion_path, save_path, scene, use_mask=True)

