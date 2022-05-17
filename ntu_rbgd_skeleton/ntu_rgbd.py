from pkgutil import get_data
import h5py as h5
import os
import torch
import numpy as np

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from PIL import Image
import io

from torch.utils.data import Dataset


class NTU_RGBD(Dataset):
    def __init__(self,
                 h5_file,
                 keys=None,
                 n_t=50,
                 batch_size=1,
                 device='cpu',
                 filter_t=False,
                 n_steps_skip=0):
        # Keys are actions
        assert os.path.exists(h5_file), f"HDF5 file {h5_file} does not exist"
        self._h5_file_handle = h5.File(h5_file, 'r')
        self.keys = keys if keys is not None else list(
            self._h5_file_handle.keys())
        print(self.keys)
        self.n_t = n_t
        max_seq_len = self._h5_file_handle[f"{self.keys[0]}/pose"].shape[1]
        self.t = torch.linspace(0., 1., min(max_seq_len, self.n_t)).to(device)

        self._key_start_idx = np.zeros(
            len(self.keys),
            dtype=np.int64)  # starting index for each key in __getitem__
        self._key_indices = {
        }  # element indices for each key in the hdf5 dataset
        n_data = 0

        for i, k in enumerate(self.keys):
            self._key_start_idx[i] = n_data
            if not filter_t:
                self._key_indices[k] = np.arange(
                    int(self._h5_file_handle[k].attrs['len']))
            else:
                seq_len = np.array(self._h5_file_handle[f"{k}/n_frames"])
                self._key_indices[k] = np.ravel(np.nonzero(seq_len >= n_t))

            n_data += len(self._key_indices[k])
        self.max_idx = n_data
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(self.max_idx / self.batch_size))
        self.device = device
        self.n_steps_skip = n_steps_skip

    def __len__(self):
        return self.max_idx

    def __getitem__(self, index):
        action, idx = self.get_action_idx(index)

        if self.n_steps_skip > 0:
            pose = torch.tensor(self._h5_file_handle[f"{action}/pose"][
                idx, ::self.n_steps_skip]).to(self.device)
            pose = pose[:self.
                        n_t]  #because we select specific idx, so first position is time
        else:
            pose = torch.tensor(
                self._h5_file_handle[f"{action}/pose"][idx, :self.n_t]).to(
                    self.device)
        return pose

    def get_action_idx(self, global_idx):
        _sub = self._key_start_idx - global_idx
        action_idx = int(np.nonzero(_sub <= 0)[0][-1])
        action = self.keys[action_idx]
        idx = self._key_indices[action][-_sub[action_idx]]
        return action, idx

    def close(self):
        self._h5_file_handle.close()

    def visualize(
            self,
            traj,
            concatenate=True,
            plot_path=None,  #'plots/rotmnist/traj',
            img_w=None,
            img_h=None,
            save_separate=False,
            add_frame=True,
            **kwargs):
        T = len(traj)
        try:
            # In case of tensor
            traj = traj.cpu().numpy()
        except:
            pass

        # traj = (traj * (self.data_max - self.data_min) +\
        #   self.data_min)

        def save_image(data, filename):

            im = Image.fromarray(np.transpose(data, (1, 0, 2)))
            #im = ImageOps.flip(im)
            if img_w is not None and img_h is not None:
                n_pics = im.size[0] // im.size[1]  #w//h
                im = im.resize((img_w * n_pics, img_h))
            im.save(filename)

        def resize_array(data):

            if img_w is not None and img_h is not None:
                if len(data.shape) == 2:
                    # Grey scale
                    im = Image.fromarray(np.transpose(data))
                else:
                    im = Image.fromarray(np.transpose(data, (1, 0, 2)))
                n_pics = im.size[0] // im.size[1]  #w//h
                im = im.resize((img_w * n_pics, img_h))
                if len(data.shape) == 2:
                    # Grey scale
                    resized_data = np.transpose(
                        self.arr_from_img(im)[0].astype(np.uint8))
                else:
                    resized_data = np.transpose(self.arr_from_img(im),
                                                (2, 1, 0)).astype(np.uint8)
                return resized_data
            else:
                return data

        def make_frame(img):
            img = img.astype(int)
            color_placement = -128
            width = int(0.01 * img.shape[1])
            gap = 0

            img_h, img_w = img.shape[:2]

            # Horizontal lines
            col_ind = list(range(0, img_w))
            row_ind = list(range(gap, width+gap)) +\
                list(range(img_h - width - gap,  img_h - gap))
            img[np.ix_(row_ind, col_ind)] = color_placement

            # Vertical lines
            col_ind = list(range(0, width))+\
                list(range(img_w-width, img_w))
            row_ind = list(range(gap, img_h - gap))
            img[np.ix_(row_ind, col_ind)] = color_placement

            RGB = np.zeros((img_h, img_w, 3))

            if len(img.shape) < 3:
                RGB[img != color_placement] = np.stack(
                    [img[img != color_placement]] * 3, axis=1)
                # Make frame different color
                RGB[(img == color_placement)] = [131, 156, 255]
            else:
                RGB[img != color_placement] = img[img != color_placement]
                # Make frame different color
                RGB[(img == color_placement)[:, :, 0]] = [131, 156, 255]

            return RGB.astype(np.uint8)  #Image.fromarray(RGB.astype(np.uint8))

        if concatenate:
            concat_image = []

        if plot_path is not None:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        for t in range(T):
            image = self.get_picture_array(traj[t][None], 0, 0)

            if add_frame:
                image = make_frame(image)
            if concatenate:
                concat_image.append(image)
            if plot_path is not None and save_separate:
                tmp = list(os.path.splitext(plot_path))
                print(tmp)
                if tmp[1] == '':
                    # Extension
                    tmp[1] = 'png'
                save_image(image, tmp[0] + '_%03d.' % t + tmp[1])

        if concatenate:
            # Concatenate to 0, because we transpose images when save.
            # Because PIL and numpy have different order of axes
            concat_image = np.concatenate(concat_image, axis=0)
            if plot_path is not None:
                tmp = list(os.path.splitext(plot_path))
                if tmp[1] == '':
                    # Extension
                    tmp[1] = '.png'
                save_image(concat_image, tmp[0] + '_concat' + tmp[1])

            concat_image = resize_array(concat_image)
            return concat_image

    def arr_from_img(self, im, shift=0):
        w, h = im.size
        arr = im.getdata()
        #c = int(np.product(arr.size) / (w * h))
        c = len(im.getbands())
        return np.asarray(arr, dtype=np.float32).reshape(
            (h, w, c)).transpose(2, 1, 0) - shift * 255

    def get_picture_array(self, X, index, shift=0):
        X = self.plot_skeleton(X[0])[None]

        ch, w, h = X.shape[1], X.shape[2], X.shape[3]
        ret = (X[index] + shift*255).\
            reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
        if ch == 1:
            ret = ret.reshape(h, w)

        return ret

    def plot_skeleton(self, x):

        xmax = np.max(x[0]) * 1.1
        xmin = np.min(x[0]) * 1.1
        ymax = np.max(x[1]) * 1.1
        ymin = np.min(x[1]) * 1.1

        x = x.transpose()  #[25, 3] -> [3, 25]
        # Determine which nodes are connected as bones according to NTU skeleton structure
        # Note that the sequence number starts from 0 and needs to be minus 1
        arms = np.array([24, 12, 11, 10, 9, 21, 5, 6, 7, 8, 22]) - 1  #Arms
        rightHand = np.array([12, 25]) - 1  #one 's right hand
        leftHand = np.array([8, 23]) - 1  #left hand
        legs = np.array([20, 19, 18, 17, 1, 13, 14, 15, 16]) - 1  #leg
        body = np.array([4, 3, 21, 2, 1]) - 1  #body

        color_joint = 'blue'  # '#03ff'  #Joint point color
        color_bone = 'red'  #Bone color

        #fig = plt.figure()  #figsize=(.28, .28))
        fig, ax = plt.subplots()  #figsize=(2, 2))
        ax.axis('off')
        #Drawing joint xs through scatter diagram
        plt.scatter(x[0, :], x[1, :], c=color_joint, s=40.0)
        #Draw the connecting line between two xs through the line diagram,
        #that is, the bone
        plt.plot(x[0, arms], x[1, arms], c=color_bone, lw=2.0)
        plt.plot(x[0, rightHand], x[1, rightHand], c=color_bone, lw=2.0)
        plt.plot(x[0, leftHand], x[1, leftHand], c=color_bone, lw=2.0)
        plt.plot(x[0, legs], x[1, legs], c=color_bone, lw=2.0)
        plt.plot(x[0, body], x[1, body], c=color_bone, lw=2.0)

        #plt.xlim(xmin, xmax)
        #plt.ylim(ymin, ymax)
        plt.xticks([], [])
        plt.yticks([], [])

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')  #, dpi=DPI)
        io_buf.seek(0)
        data_plot = np.reshape(np.frombuffer(io_buf.getvalue(),
                                             dtype=np.uint8),
                               newshape=(int(fig.bbox.bounds[3]),
                                         int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        plt.close()
        data_plot = data_plot[:, :, :3].transpose(2, 0, 1)
        return data_plot

        #data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)


if __name__ == "__main__":
    dataset_1 = NTU_RGBD("./nturgb+d.hdf5", n_t=100)
    dataset_2 = NTU_RGBD("./nturgb+d.hdf5", n_t=10, n_steps_skip=10)

    print(len(dataset_1))
    print(dataset_1[0].shape)
    print(len(dataset_2))
    print(dataset_2[0].shape)

    from tqdm import tqdm

    [dataset_1[i] for i in tqdm(range(len(dataset_1)))]
    [dataset_2[i] for i in tqdm(range(len(dataset_2)))]
