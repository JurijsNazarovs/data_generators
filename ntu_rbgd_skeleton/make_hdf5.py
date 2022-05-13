from ast import arg
import h5py as h5
from glob import glob
import numpy as np
import argparse
import os

import re

N_JOINTS = 25
N_BODY = 1
MAX_FRAMES = 100
MAX_ACTIONS = 120

def parse_skeleton_file(file):
    skeleton_xyz = np.zeros((MAX_FRAMES, N_JOINTS, 3))
    with open(file, 'r') as f:
        n_frames = int(f.readline())
        for i in range(min(MAX_FRAMES, n_frames)):
            n_body = int(f.readline())
            body_info = [float(x) for x in f.readline().split(' ')]
            n_joints = int(f.readline())

            if n_body != N_BODY or n_joints != N_JOINTS:
                raise RuntimeError("# bodies or joints mismatch with preset value")

            for j in range(n_joints):
                skeleton_xyz[i, j] = np.array(f.readline().split(' ')[:3])
    
    return skeleton_xyz, min(MAX_FRAMES, n_frames)

def make_hdf5(data_root, h5_target):
    pattern = "(S[0-9]{3})C[0-9]{3}P[0-9]{3}R[0-9]{3}(A[0-9]{3}).skeleton"

    with h5.File(h5_target, 'w') as h5f:
        groups = {f"A{a_id + 1:03d}": h5f.create_group(f"A{a_id + 1:03d}") for a_id in range(MAX_ACTIONS)}
        datasets = {
            k: {
                    "pose": groups[k].create_dataset("pose", 
                                                    (100, MAX_FRAMES, N_JOINTS, 3), 
                                                    chunks=(100, MAX_FRAMES, N_JOINTS, 3), 
                                                    maxshape=(None, MAX_FRAMES, N_JOINTS, 3),
                                                    dtype='f4'),
                    "n_frames": groups[k].create_dataset("n_frames", 
                                                    (100,), 
                                                    chunks=(100,), 
                                                    maxshape=(None,),
                                                    dtype='u4')
                }
                for k in groups
        }
        dataset_len = {k: 0 for k in groups}
        for file in glob(f"{data_root}/*.SKELETON"):
            print(f"Parsing {os.path.basename(file)}")
            matched = re.match(pattern, os.path.basename(file))
            subject_id, action_id = matched.group(1, 2)
            try:
                skeleton_xyz, n_frames = parse_skeleton_file(file)

                datasets[action_id]["pose"][dataset_len[action_id]] = skeleton_xyz
                datasets[action_id]["n_frames"][dataset_len[action_id]] = n_frames
                dataset_len[action_id] += 1
            except:
                pass
        
        for k in groups:
            groups[k].attrs['len'] = dataset_len[k]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", "-d", default="./nturgb+d_skeletons", type=str)
    parser.add_argument("--output-file", "-o", default="./nturgb+d.hdf5")

    args = parser.parse_args()

    make_hdf5(args.data_root, args.output_file)

