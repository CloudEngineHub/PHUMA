'''
    Download LAFAN1 and LocoMuJoCo datasets
'''
import os
import argparse
import subprocess
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Download LAFAN1 and LocoMuJoCo datasets")
    parser.add_argument('--project_dir', type=str, required=True, help="Path to the project directory")
    parser.add_argument('--remove_original', action='store_true', help="Remove original datasets")
    return parser.parse_args()


def run_command(cmd, shell=True):
    """Run a shell command and print output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, check=True)
    return result


def main(args):
    project_dir = args.project_dir
    
    # Download LAFAN1 dataset
    lafan_dir = os.path.join(project_dir, 'data', 'original_datasets', 'LAFAN1')
    print(f'Downloading LAFAN1 dataset to {lafan_dir}')
    os.makedirs(lafan_dir, exist_ok=True)
    run_command(f'huggingface-cli download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset --local-dir {lafan_dir}')
    
    # Remove h1 dataset
    run_command(f'rm -rf {os.path.join(lafan_dir, "h1")}')
    
    # Download LocoMuJoCo G1 dataset
    loco_g1_dir = os.path.join(project_dir, 'data', 'original_datasets', 'loco_mujoco', 'g1')
    print(f'Downloading LocoMuJoCo dataset to {loco_g1_dir}')
    os.makedirs(loco_g1_dir, exist_ok=True)
    run_command(f'huggingface-cli download robfiras/loco-mujoco-datasets --repo-type dataset --local-dir {loco_g1_dir} --include "DefaultDatasets/mocap/UnitreeG1/*"')
    
    # Move G1 files to the correct location
    g1_source_dir = os.path.join(loco_g1_dir, 'DefaultDatasets', 'mocap', 'UnitreeG1')
    if os.path.exists(g1_source_dir):
        for item in os.listdir(g1_source_dir):
            src = os.path.join(g1_source_dir, item)
            dst = os.path.join(loco_g1_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(os.path.join(loco_g1_dir, 'DefaultDatasets'))
    
    # Download LocoMuJoCo H1_2 dataset
    loco_h1_dir = os.path.join(project_dir, 'data', 'original_datasets', 'loco_mujoco', 'h1_2')
    print(f'Downloading LocoMuJoCo dataset to {loco_h1_dir}')
    os.makedirs(loco_h1_dir, exist_ok=True)
    run_command(f'huggingface-cli download robfiras/loco-mujoco-datasets --repo-type dataset --local-dir {loco_h1_dir} --include "DefaultDatasets/mocap/UnitreeH1v2/*"')
    
    # Move H1_2 files to the correct location
    h1_source_dir = os.path.join(loco_h1_dir, 'DefaultDatasets', 'mocap', 'UnitreeH1v2')
    if os.path.exists(h1_source_dir):
        for item in os.listdir(h1_source_dir):
            src = os.path.join(h1_source_dir, item)
            dst = os.path.join(loco_h1_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(os.path.join(loco_h1_dir, 'DefaultDatasets'))
    
    # Preprocess LAFAN1 dataset
    preprocessed_dir = os.path.join(project_dir, 'data', 'preprocessed_original')
    print(f'Preprocessing LAFAN1 dataset to {preprocessed_dir}')
    run_command(f'python src/utils/lafan_csv_to_npy.py --lafan_dir {lafan_dir} --output_dir {preprocessed_dir}')
    
    # Preprocess LocoMuJoCo dataset
    loco_dir = os.path.join(project_dir, 'data', 'original_datasets', 'loco_mujoco')
    print(f'Preprocessing LocoMuJoCo dataset to {preprocessed_dir}')
    run_command(f'python src/utils/locomujoco_npz_to_npy.py --locomujoco_dir {loco_dir} --output_dir {preprocessed_dir}')
    
    # Remove original datasets
    if args.remove_original:
        run_command(f'rm -rf {os.path.join(project_dir, "data", "original_datasets")}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
