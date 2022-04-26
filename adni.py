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

### DIMENSION OF 3D VOLUME
dX = 121
dY = 145
dZ = 121


class Adni(IterableDataset):
    def __init__(self,
                 datapath,
                 is_add_channel=True,
                 name="ADNI",
                 device='cpu',
                 batch_size=1,
                 is_train=False):

        self.datapath = datapath
        if Path(self.datapath).exists():
            pass
        else:
            print('ADNI path specified does not exist.')
            sys.exit(0)

        self.n_t = 3  # 3 timepoints HARDCODED HERE
        self.device = device
        self.batch_size = batch_size
        self.vectdim = int(dX * dY * dZ)  # vectorized size
        self.dim = (dX, dY, dZ)
        self.is_train = is_train  #training generator or testing
        self.is_add_channel = is_add_channel

        self.tot_samples = len(os.listdir(self.datapath))
        self.train_size = 400
        self.test_size = self.tot_samples - self.train_size
        #self.n_train_batches = self.train_size // self.batch_size
        #self.n_test_batches = self.test_size // self.batch_size
        self.n_train_batches = int(np.ceil(self.train_size / self.batch_size))
        self.n_test_batches = int(np.ceil(self.test_size / self.batch_size))

        #np.random.permutation(self.tot_samples)
        self.idxs = range(0, self.tot_samples)
        self.train_idxs = self.idxs[:self.train_size]
        self.test_idxs = self.idxs[self.train_size:]

        # each file in ADNI folder should be a separate subject
        if self.tot_samples < 1:
            print('Dataset not found.' +
                  'If download=True, will attempt to download it.')
            if download:
                raise NotImplementedError
                self._download()
            else:
                raise "Fail to obtain dataset ADNI"
        else:
            print("Loading existing data")

        t = np.arange(0, self.n_t)  # / (self.n_t - 1)
        t = t.astype(np.float32)
        self.t = torch.tensor(t).to(device)

    def __len__(self):
        return self.tot_samples

    def __iter__(self):
        self.batch_idx = 0
        return self

    def __next__(self):
        # if self.n <= self.max:
        #     result = 2 ** self.n
        #     self.n += 1
        #     return result
        # else:
        #     raise StopIteration
        batch = self._getbatch(self.batch_idx)
        if batch.shape[0] == 0:
            raise StopIteration
        else:
            self.batch_idx += 1
        return batch

    def _getbatch(self, batch_idx):
        if self.is_train:
            inds = self.train_idxs
        else:
            inds = self.test_idxs
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(inds))

        fnames = [
            "%s/%04d.npy" % (self.datapath, idx)
            for idx in inds[start_idx:end_idx]
        ]
        X_batch = self.load_data(fnames=fnames)
        if self.device is not None:
            X_batch = torch.tensor(X_batch,
                                   requires_grad=False).to(self.device)

        #return X_batch[:, :-1, :, :, :], X_batch[:, -1, :, :, :] #x, y
        #import pdb
        #pdb.set_trace()
        #hui = torch.unbind(X_batch, 0)
        return X_batch

    def _download(self):
        print("Downloading the dataset ...")
        os.makedirs(self.data_folder, exist_ok=True)
        url = "????"
        download_url(url, self.data_folder, self.data_name, None)

    def load_data(self, fpath=None, fnames=None):
        if fpath is not None:
            fnames = [
                join(fpath, f) for f in listdir(fpath)
                if isfile(join(fpath, f))
            ]

        if self.is_add_channel:
            dat = np.empty((len(fnames), self.n_t, 1, dX, dY, dZ),
                           dtype=np.float32)
        else:
            dat = np.empty((len(fnames), self.n_t, dX, dY, dZ),
                           dtype=np.float32)
        for f, i in zip(fnames, range(0, len(fnames))):
            tmp = np.load(f)

            # add channel
            if self.is_add_channel:
                shape = list(tmp.shape)
                shape.insert(-3, 1)
                tmp = tmp.reshape(shape)

            dat[i] = tmp
        return dat

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
