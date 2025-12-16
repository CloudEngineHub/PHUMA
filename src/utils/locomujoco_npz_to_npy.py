import os
import argparse
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LocoMuJoCo NPZ files to NPY format")
    parser.add_argument('--locomujoco_dir', type=str, required=True, help="Path to the LocoMuJoCo dataset directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory")
    return parser.parse_args()


def main(args):
    
    locomujoco_dir = args.locomujoco_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(locomujoco_dir):
        for file in tqdm(files, desc=f"Processing files in {root.split('/')[-1]}"):
            robot_type = root.split('/')[-1]
            if file.endswith('.npz'):
                data = np.load(os.path.join(root, file), allow_pickle=True)
                motion_data = data['qpos']
                save_dir = os.path.join(output_dir, root.split('/')[-1], 'LocoMuJoCo')
                os.makedirs(save_dir, exist_ok=True)

                N = motion_data.shape[0]
                
                root_trans = motion_data[:,:3]
                root_ori = motion_data[:,3:7][:, [1, 2, 3, 0]]
                dof_pos = motion_data[:,7:]
                
                if robot_type == 'g1':
                    dof_pos = np.concatenate([
                        dof_pos[:, :12], # lower body
                        dof_pos[:, 12][:, None], np.zeros((N, 2)), # waist
                        dof_pos[:, 13:17], # left arm
                        dof_pos[:, 18:22], # right arm
                    ], axis=1)
                
                elif robot_type == 'h1_2':
                    dof_pos = np.concatenate([
                        dof_pos[:, :12], # lower body
                        dof_pos[:, 12][:, None], # waist
                        dof_pos[:, 13:17], np.zeros((N, 3)), # left arm
                        dof_pos[:, 17:21], np.zeros((N, 3)), # right arm
                    ], axis=1)
                
                motion = {
                    'root_trans': root_trans.astype(np.float32),
                    'root_ori': root_ori.astype(np.float32),
                    'dof_pos': dof_pos,
                    'fps': 30
                }
                np.save(os.path.join(save_dir, file.replace('.npz', '.npy')), motion)


if __name__ == '__main__':
    args = parse_args()
    main(args)
