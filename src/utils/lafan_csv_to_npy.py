import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LaFAN1 CSV files to NPY format")
    parser.add_argument('--lafan_dir', type=str, required=True, help="Path to the LaFAN1 dataset directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory")
    return parser.parse_args()


def main(args):    
    lafan_dir = args.lafan_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(lafan_dir):
        for file in tqdm(files, desc=f"Processing files in {root.split('/')[-1]}"):
            robot_type = root.split('/')[-1]
            if file.endswith('.csv'):
                sub_folder = file.split('_')[0]
                file_name = file.split('_')[-1]
                df = pd.read_csv(os.path.join(root, file))
                save_dir = os.path.join(output_dir, root.split('/')[-1], 'LAFAN1', sub_folder)
                os.makedirs(save_dir, exist_ok=True)

                root_trans_list = []
                root_ori_list = []
                dof_pos_list = []
                for idx, row in tqdm(df.iterrows(), desc="Processing"):
                    # row is a Series object, so we can access each column value
                    # Get all values as a list
                    values = row.values.tolist()
                    root_trans = values[0:3]
                    root_ori = values[3:7]
                    dof_pos = values[7:]
                    
                    if robot_type == 'g1':
                        dof_pos = dof_pos[:19] + dof_pos[22:26]
                        
                    root_trans_list.append(root_trans)
                    root_ori_list.append(root_ori)
                    dof_pos_list.append(dof_pos)
                    
                motion = {
                    'root_trans': np.array(root_trans_list, dtype=np.float32),
                    'root_ori': np.array(root_ori_list, dtype=np.float32),
                    'dof_pos': np.array(dof_pos_list),
                    'fps': 30
                }
                np.save(os.path.join(save_dir, file_name.replace('.csv', '.npy')), motion)


if __name__ == '__main__':
    args = parse_args()
    main(args)
