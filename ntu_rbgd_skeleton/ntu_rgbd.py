from pkgutil import get_data
import h5py as h5
import os
import torch
import numpy as np

from torch.utils.data import Dataset

class NTU_RGBD(Dataset):
    def __init__(self, h5_file, keys=None, max_len=50):
        assert os.path.exists(h5_file), f"HDF5 file {h5_file} does not exist"
        self._h5_file_handle = h5.File(h5_file, 'r')
        self.keys = keys if keys is not None else list(self._h5_file_handle.keys())
        self.max_len = max_len
        data_seq_len = self._h5_file_handle[f"{self.keys[0]}/pose"].shape[1]
        self.t = torch.linspace(0., 1., min(data_seq_len, self.max_len))
        
        self._key_start_idx = np.zeros(len(self.keys), dtype=np.int64)
        sum = 0
        for i, k in enumerate(self.keys):
            self._key_start_idx[i] = sum
            sum += self._h5_file_handle[k].attrs['len']
        self.max_idx = sum
    
    def __getitem__(self, index):
        action, idx = self.get_action_idx(index)
        
        pose = torch.tensor(self._h5_file_handle[f"{action}/pose"][idx, :self.max_len])

        return pose
    
    def __len__(self):
        return self.max_idx

    def get_action_idx(self, global_idx):
        _sub = self._key_start_idx - global_idx
        action_idx = int(np.nonzero(_sub >= 0)[0][0])
        action = self.keys[action_idx]
        idx = _sub[action_idx]
        return action, idx
    
    def close(self):
        self._h5_file_handle.close()
    
if __name__ == "__main__":
    dataset = NTU_RGBD("../../func_ode/data/nturgb+d.hdf5", max_len=150)
    print(len(dataset.t))
    print(dataset[500])