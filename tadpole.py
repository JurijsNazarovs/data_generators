import os
import numpy as np
import torch
import lib.utils as utils
from torchvision.datasets.utils import download_url
from PIL import Image
from scipy import ndimage
from pathlib import Path

from os import listdir
from os.path import isfile, join
from torch.utils.data import IterableDataset

from lib.adni.visualizer3d import Visualizer as vis3d
from scipy.io import loadmat


class Tadpole(IterableDataset):
    def __init__(self, datapath, name="TADPOLE", device='cpu'):

        self.datapath = datapath
        if Path(self.datapath).exists():
            pass
        else:
            print('TADPOLE path specified does not exist.')
            sys.exit(0)

        self.n_t = 3  # 3 timepoints HARDCODED HERE
        self.device = device

        # each file in ADNI folder should be a separate subject
        self.data_file = self.datapath + '/x.mat'
        if not os.path.exists(self.data_file):
            print('Dataset not found.' +
                  'If download=True, will attempt to download it.')
            if download:
                raise NotImplementedError
                self._download()
            else:
                raise "Fail to obtain dataset TADPOLE"
        else:
            print("Loading existing data")

        data = loadmat(self.data_file)['X']
        self.data = torch.Tensor(data).to(device)
        t = np.arange(0, self.n_t) / (self.n_t - 1)
        t = t.astype(np.float32)
        self.t = torch.tensor(t).to(device)

    def _download(self):
        print("Downloading the dataset ...")
        os.makedirs(self.data_folder, exist_ok=True)
        url = "????"
        download_url(url, self.data_folder, self.data_name, None)

    def visualize(self, traj, plot_path=None, slices=(50, 60, 50), **kwargs):
        try:
            traj = traj.cpu().numpy()
        except:
            pass

        frame_size = np.array(
            (traj.shape[-3], traj.shape[-1]))  # size of image
        #frame_size = np.array((10, 10 * 2))  # size of image

        # Plot x, y, prediction, abs(difference)
        frame = vis3d(frame_size=frame_size,
                      slices=slices,
                      ncols=traj.shape[0],
                      figsize=(10, 10))

        frame.make_plot(traj)
        if plot_path is not None:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            frame.saveIt(path=plot_path)

        # to return image as np array
        # frame.fig.canvas.draw()
        # w, h = frame.fig.canvas.get_width_height()
        # buf = np.fromstring(frame.fig.canvas.tostring_argb(), dtype=np.uint8)
        # buf.shape = (w, h, 4)
        # return buf
        return None

    def arr_from_img(self, im, shift=0):
        w, h = im.size
        arr = im.getdata()
        c = int(np.product(arr.size) / (w * h))
        return np.asarray(arr, dtype=np.float32).reshape(
            (h, w, c)).transpose(2, 1, 0) - shift * 255

    def get_picture_array(self, X, index, shift=0):
        ch, w, h = X.shape[1], X.shape[2], X.shape[3]
        ret = (X[index] + shift*255).\
            reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
        if ch == 1:
            ret = ret.reshape(h, w)
        return ret

    def _check_exists(self):
        return os.path.exists(self.data_file)

    @property
    def data_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    # def __getitem__(self, index):
    #     return self.data[index]

    def get_dataset(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def size(self, ind=None):
        if ind is not None:
            return self.data.shape[ind]
        return self.data.shape

    def __repr__(self):
        s = ('data_file = {data_file}', 'n_samples={n_samples}', 'n_t = {n_t}',
             'fix_eff = {fix_eff}', 'rand_eff_std = {rand_eff_std}')
        return '\n'.join(s).format(**self.__dict__)
