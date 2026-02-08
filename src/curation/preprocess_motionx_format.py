import os
import argparse
from tqdm import tqdm
import numpy as np

from utils.motionx_npz_to_npy import convert_npz_to_npy

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Motion-X (N,322) npy or npz to PHUMA (N,69)")
    parser.add_argument("--human_pose_folder", required=True)
    parser.add_argument("--output_dir", type=str, default="data/human_pose")
    parser.add_argument("--target_fps", type=int, default=30, help="Target FPS for npz downsampling (default: 30)")
    return parser.parse_args()


def convert_322_to_69(human_pose_motionx):
    """Convert Motion-X (N, 322) format to PHUMA (N, 69) format."""
    return np.concatenate([
        human_pose_motionx[:, 309:309+3],  # transl: (N, 3)
        human_pose_motionx[:, 0:0+3],      # global_orient: (N, 3)
        human_pose_motionx[:, 3:3+63]      # body_pose: (N, 63)
    ], axis=1)  # Shape: (N, 69)


def main(args):
    converted, skipped, errors = 0, 0, 0

    for root, dirs, files in os.walk(args.human_pose_folder):
        for motion_file in tqdm(files, desc="Processing"):
            is_npy = motion_file.endswith('.npy')
            is_npz = motion_file.endswith('.npz')
            if not is_npy and not is_npz:
                continue

            try:
                motion_file_path = os.path.join(root, motion_file)

                if is_npz:
                    # Convert npz -> 322-dim npy using motionx_npz_to_npy
                    human_pose_motionx = convert_npz_to_npy(motion_file_path, args.target_fps)
                    out_name = motion_file.replace('.npz', '.npy')
                else:
                    human_pose_motionx = np.load(motion_file_path)
                    out_name = motion_file

                if human_pose_motionx.shape[1] != 322:
                    skipped += 1
                    continue

                human_pose_phuma = convert_322_to_69(human_pose_motionx)

                # Get relative path from parent of human_pose_folder
                parent_dir = os.path.dirname(args.human_pose_folder)
                rel_path = os.path.relpath(root, parent_dir)
                output_path = os.path.join(args.output_dir, rel_path, out_name)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, human_pose_phuma)
                converted += 1

            except Exception as e:
                print(f"\nError: {motion_file_path} - {e}")
                errors += 1

    print(f"\nDone: {converted} converted, {skipped} skipped, {errors} errors")

if __name__ == '__main__':
    args = parse_args()
    main(args)