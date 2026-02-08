import os
import argparse
from tqdm import tqdm
import numpy as np
import torch


def axis_angle_to_matrix(axis_angle):
    """Convert axis-angle to rotation matrix."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    K = torch.zeros(axis_angle.shape[:-1] + (3, 3), device=axis_angle.device)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    
    eye = torch.eye(3, device=axis_angle.device).expand(axis_angle.shape[:-1] + (3, 3))
    R = eye + sin_angle.unsqueeze(-1) * K + (1 - cos_angle.unsqueeze(-1)) * torch.matmul(K, K)
    
    return R


def matrix_to_axis_angle(matrix):
    """Convert rotation matrix to axis-angle."""
    angle = torch.acos(torch.clamp((matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2] - 1) / 2, -1, 1))
    
    axis = torch.stack([
        matrix[..., 2, 1] - matrix[..., 1, 2],
        matrix[..., 0, 2] - matrix[..., 2, 0],
        matrix[..., 1, 0] - matrix[..., 0, 1]
    ], dim=-1)
    
    axis = axis / (2 * torch.sin(angle).unsqueeze(-1) + 1e-8)
    
    return axis * angle.unsqueeze(-1)


def compute_canonical_transform(global_orient):
    """Transform global orientation from Y-up to Z-up coordinate system."""
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient


def transform_translation(trans):
    """Transform translation from Y-up to Z-up coordinate system."""
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
    trans[:, 2] = trans[:, 2] * (-1)
    return trans


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Motion-X stageii.npz files to 322-dim npy format")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the directory containing npz files")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory")
    parser.add_argument('--target_fps', type=int, default=30, help="Target FPS for downsampling (default: 30)")
    return parser.parse_args()


def convert_npz_to_npy(npz_path, target_fps=30):
    """
    Convert Motion-X stageii.npz to 322-dim npy format.
    
    npz keys:
        - root_orient: (N, 3)
        - pose_body: (N, 63)
        - pose_hand: (N, 90)
        - pose_jaw: (N, 3)
        - trans: (N, 3)
        - betas: (16,) or (10,)
        - mocap_frame_rate: float
    
    npy output: (N, 322) array
        - [:, 0:3] root_orient
        - [:, 3:66] pose_body
        - [:, 66:156] pose_hand
        - [:, 156:159] pose_jaw
        - [:, 159:209] face_expression (zeros)
        - [:, 209:309] face_shape (zeros)
        - [:, 309:312] trans
        - [:, 312:322] betas
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Get frame rate and compute downsampling factor
    if 'mocap_frame_rate' in data:
        fps = float(data['mocap_frame_rate'])
    elif 'mocap_framerate' in data:
        fps = float(data['mocap_framerate'])
    else:
        fps = target_fps  # Assume already at target fps
    
    down_sample = max(1, int(fps / target_fps))
    
    # Get frames with downsampling
    root_orient = data['root_orient'][::down_sample]  # (N, 3)
    pose_body = data['pose_body'][::down_sample]      # (N, 63)
    pose_hand = data['pose_hand'][::down_sample]      # (N, 90)
    pose_jaw = data['pose_jaw'][::down_sample]        # (N, 3)
    trans = data['trans'][::down_sample]              # (N, 3)
    betas = data['betas'][:10]                        # (10,)
    
    N = root_orient.shape[0]
    
    # Apply coordinate system transformation (Y-up to Z-up)
    root_orient = compute_canonical_transform(torch.from_numpy(root_orient)).numpy()
    trans = transform_translation(trans)
    
    # Concatenate to 322-dim format
    pose = np.concatenate([
        root_orient,                    # (N, 3)
        pose_body,                      # (N, 63)
        pose_hand,                      # (N, 90)
        pose_jaw,                       # (N, 3)
        np.zeros((N, 50)),              # face_expression (N, 50)
        np.zeros((N, 100)),             # face_shape (N, 100)
        trans,                          # (N, 3)
        np.tile(betas, (N, 1))          # betas (N, 10)
    ], axis=1)  # (N, 322)
    
    return pose.astype(np.float32)


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    converted, skipped, errors = 0, 0, 0
    
    for root, dirs, files in os.walk(input_dir):
        npz_files = [f for f in files if f.endswith('.npz')]
        
        for file in tqdm(npz_files, desc=f"Processing {os.path.basename(root)}"):
            try:
                input_path = os.path.join(root, file)
                
                # Preserve relative directory structure
                rel_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, rel_path)
                os.makedirs(save_dir, exist_ok=True)
                
                output_path = os.path.join(save_dir, file.replace('.npz', '.npy'))
                
                # Convert and save
                pose = convert_npz_to_npy(input_path, args.target_fps)
                np.save(output_path, pose)
                converted += 1
                
            except Exception as e:
                print(f"\nError: {file} - {e}")
                errors += 1
    
    print(f"\nDone: {converted} converted, {skipped} skipped, {errors} errors")


if __name__ == '__main__':
    args = parse_args()
    main(args)
