import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import mujoco
import yaml
from scipy.spatial import ConvexHull, Delaunay
from utils.smpl import _point_to_segment_dist


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess H1_2 folder")
    parser.add_argument("--project_dir", required=True)
    parser.add_argument('--output_dir', type=str, help="Path to the output directory")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--verbose", type=int, default=1)
    
    parser.add_argument("--ground_threshold", type=float, default=0.05)
    parser.add_argument("--min_com_height_threshold", type=float, default=0.6)
    parser.add_argument("--com_to_bos_distance_threshold", type=float, default=0.24)

    parser.add_argument("--chunk_size", type=int, default=4 * 30)
    parser.add_argument("--chunk_overlap", type=int, default=0.5 * 30)
    parser.add_argument("--chunk_min_frames", type=int, default=1 * 30)
    return parser.parse_args()


def set_qpos(root_trans, root_ori, dof_pos):
    qpos = np.zeros(len(root_trans) + len(root_ori) + len(dof_pos))
    qpos[0: 3] = root_trans
    qpos[3: 7] = root_ori[..., [3, 0, 1, 2]]
    qpos[7:  ] = dof_pos

    return qpos


def get_ground_offest(
    chunk_root_trans, 
    chunk_root_ori, 
    chunk_dof_pos, 
    foot_body_indices, 
    ground_threshold,
    humanoid_model,
    data):
    
    data.qpos = set_qpos(chunk_root_trans[0], chunk_root_ori[0], chunk_dof_pos[0])
    mujoco.mj_resetData(humanoid_model, data)
    
    N = chunk_root_trans.shape[0]
    
    pred_contact_heights = []
    for i in range(N):
        data.qpos = set_qpos(chunk_root_trans[i], chunk_root_ori[i], chunk_dof_pos[i])
        mujoco.mj_forward(humanoid_model, data)
        body_trans = data.xpos[1:]
        foot_trans = body_trans[foot_body_indices] # shape: (4, 3) for left toe, left heel, right toe, right heel
        foot_heights = foot_trans[:, 2] 
        pred_contact_heights.append(foot_heights.min())
    
    max_count = 0
    best_h_stars = []
    heights = np.array(pred_contact_heights)
    heights.sort()  # Sort heights in ascending order
    
    for h_window_start in heights:
        h_window_end = h_window_start + ground_threshold
        in_window = (heights >= h_window_start) & (heights <= h_window_end)  # Boolean mask: heights that lie within [h_window_start, h_window_end]
        current_count = int(in_window.sum())  # Number of samples inside the current window
        current_h_star = h_window_start + ground_threshold / 2.0
        if current_count > max_count:
            max_count = current_count
            best_h_stars = [current_h_star]
        elif current_count == max_count:
            best_h_stars.append(current_h_star)
    return np.median(best_h_stars)


def calculate_bos_distance(base_of_support_xy, com_xy):
    N = base_of_support_xy.shape[0]
    frame_distances = []
    
    for i in range(N):
        unique_points = np.unique(base_of_support_xy[i], axis=0)
        
        hull = ConvexHull(unique_points)
        if Delaunay(hull.points[hull.vertices]).find_simplex(com_xy[i]) >= 0:
            dist = 0.0
        else:
            hull_vertices = hull.points[hull.vertices]
            min_dist_to_edge = float('inf')
            for j in range(len(hull_vertices)):
                p1 = hull_vertices[j]
                p2 = hull_vertices[(j + 1) % len(hull_vertices)]
                seg_dist = _point_to_segment_dist(com_xy[i], p1, p2)
                if seg_dist < min_dist_to_edge:
                    min_dist_to_edge = seg_dist
            dist = min_dist_to_edge
                
        frame_distances.append(dist)
        
    return np.mean(frame_distances) if sum(frame_distances) > 0.0 else 0.0


def main(args):
    humanoid_pose_folder = os.path.join(args.project_dir, 'data', 'preprocessed_original', 'h1_2')
    humanoid_model_dir = os.path.join(args.project_dir, "asset", "humanoid_model", 'h1_2')
    scene_path = os.path.join(humanoid_model_dir, "scene.xml")
    humanoid_model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(humanoid_model)
    config_path = os.path.join(humanoid_model_dir, "config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Use model (not data) to query number of bodies and their names
    body_names = [humanoid_model.body(i).name for i in range(1, humanoid_model.nbody)]
    foot_body_indices = [body_names.index(body) for body in body_names if 'toe' in body or 'heel' in body]
    
    
    data.qpos = set_qpos(np.array(config['root_pos']), np.array(config['root_ori']), np.array(config['dof_pos']))
    mujoco.mj_resetData(humanoid_model, data)
    
    for root, dirs, files in os.walk(humanoid_pose_folder):
        for motion_file in tqdm(files, desc=f"Processing files in {root.split('/')[-1]}"):
            if not motion_file.endswith('.npy'):
                continue
            
            motion_file_path = os.path.join(root, motion_file)
            motion_data = np.load(motion_file_path, allow_pickle=True).item()
            
            N = motion_data['root_trans'].shape[0]
            root_trans = motion_data['root_trans']
            root_ori = motion_data['root_ori']
            dof_pos = motion_data['dof_pos']
            fps = motion_data['fps']
            
            start_frame = 0
            chunk_id = 0

            # Get base filename without extension
            base_filename = os.path.splitext(motion_file)[0]
            
            while start_frame < N:
                chunk_file = f"{base_filename}_chunk_{chunk_id:04d}"

                end_frame = start_frame + args.chunk_size
                if end_frame + (args.chunk_min_frames - args.chunk_overlap) > N:
                    end_frame = N
                else:
                    end_frame = min(end_frame, N)
                    
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                chunk_root_trans = root_trans[start_frame:end_frame].copy()
                chunk_root_ori = root_ori[start_frame:end_frame].copy()
                chunk_dof_pos = dof_pos[start_frame:end_frame].copy()
                
                N_chunk = chunk_root_trans.shape[0]
                if N_chunk < args.chunk_min_frames:
                    break
                
                is_valid = True
                
                folder = os.path.relpath(root, humanoid_pose_folder).split('/')[0]
                
                # Ground Offset
                if folder == 'LocoMuJoCo':
                    pred_ground_offset = get_ground_offest(
                        chunk_root_trans, 
                        chunk_root_ori, 
                        chunk_dof_pos, 
                        foot_body_indices, 
                        args.ground_threshold, 
                        humanoid_model, 
                        data)
                    
                    # Subtract ground offset from root trans
                    chunk_root_trans[:, 2] -= pred_ground_offset
                
                com_positions = []
                base_of_support_xy = []
                for i in range(N_chunk):
                    data.qpos = set_qpos(chunk_root_trans[i], chunk_root_ori[i], chunk_dof_pos[i])
                    mujoco.mj_forward(humanoid_model, data)
                    
                    com_position = data.subtree_com[0].copy()
                    body_trans = data.xpos[1:]
                    foot_trans = body_trans[foot_body_indices] # shape: (4, 3) for left toe, left heel, right toe, right heel
                    foot_xy = foot_trans[:, :2].copy()
                    
                    com_positions.append(com_position)
                    base_of_support_xy.append(foot_xy)
                    
                com_positions = np.array(com_positions)
                base_of_support_xy = np.stack(base_of_support_xy, axis=0)
                
                # Com Height
                min_com_height = com_positions[:, 2].min()
                if min_com_height < args.min_com_height_threshold:
                    if args.verbose:
                        print(f"[FILTER OUT] File: {chunk_file} - Min com height {min_com_height} lower than {args.min_com_height_threshold}.")
                    is_valid = False
                
                # Com to BoS Distance
                com_xy = com_positions[:, :2]
                com_to_bos_distance = calculate_bos_distance(base_of_support_xy, com_xy)
                if com_to_bos_distance > args.com_to_bos_distance_threshold:
                    if args.verbose:
                        print(f"[FILTER OUT] File: {chunk_file} - Com to BoS distance {com_to_bos_distance} higher than {args.com_to_bos_distance_threshold}.")
                    is_valid = False
                    
                if is_valid:
                    rel_path = os.path.relpath(root, humanoid_pose_folder)
                    if args.output_dir is None:
                        output_dir = os.path.join(args.project_dir, 'data', 'h1_2')
                    else:
                        output_dir = args.output_dir
                    output_root = f"{output_dir}/{rel_path}"
                    
                    os.makedirs(output_root, exist_ok=True)

                    chunk_motion_data = {
                        'root_trans': chunk_root_trans,
                        'root_ori': chunk_root_ori,
                        'dof_pos': chunk_dof_pos,
                        'fps': fps,
                    }
                    
                    np.save(os.path.join(output_root, f"{chunk_file}.npy"), chunk_motion_data)

                start_frame += (args.chunk_size - args.chunk_overlap)
                chunk_id += 1
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
