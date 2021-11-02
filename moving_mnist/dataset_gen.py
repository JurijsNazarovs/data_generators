import numpy as np
from pathlib import Path
import shutil  # to remove nonempty directory
import os

import data_generators.movingmnist_gen as mmg
import importlib
importlib.reload(mmg)


class DataSetGenerator(object):
    def __init__(self,
                 history_size,
                 train_n_samples=1,
                 out_path="./data",
                 test_prop=0.1,
                 valid_prop=0,
                 y_length=1,
                 frame_size=30,
                 digit_size=10,
                 speed=10,
                 digits_p_img=1,
                 is_same_digit=True,
                 mnist_data_path=None,
                 scale=True,
                 normalize=True,
                 **kwargs):

        # Data set parameters
        self.history_size = history_size
        self.y_length = y_length  # number of predicted elements

        self.train_size = train_n_samples
        self.test_size = int(np.ceil(test_prop * train_n_samples))
        self.valid_size = int(np.ceil(valid_prop * train_n_samples))

        self.out_path = out_path + "/"
        self.train_path = self.out_path + "train/"
        self.test_path = self.out_path + "test/"
        self.valid_path = self.out_path + "valid/"

        self.scale = scale
        self.normalize = normalize

        self.norm_mean = 0.1307
        self.norm_std = 0.3081

        if not self.scale:
            self.norm_mean *= 255
            self.norm_std *= np.sqrt(255)

        # Moving mnist generator
        if is_same_digit:
            # Have to generate 1 sample of enough length, to restrict images with
            # the same type of digit
            n_samples = 1
            mm_sample_size = (self.train_size + self.test_size + self.valid_size) *\
                (self.history_size + self.y_length)
        else:
            n_samples = self.train_size + self.test_size + self.valid_size
            mm_sample_size = self.history_size + self.y_length

        self.mmGen = mmg.MovingMnistGenerator(out_path=self.out_path +
                                              "mnist/",
                                              n_samples=n_samples,
                                              sample_size=mm_sample_size,
                                              digit_size=digit_size,
                                              frame_size=frame_size,
                                              speed=speed,
                                              digits_p_img=digits_p_img,
                                              mnist_data_path=mnist_data_path,
                                              scale=scale,
                                              normalize=normalize,
                                              norm_mean=self.norm_mean,
                                              norm_std=self.norm_std)

    def generate(self, shuffle=True):
        # Generates data and returns train, test, validation sets
        if (self.train_size <= 0):
            return

        self.mmGen.generate()
        data = self.mmGen.load_data().\
            reshape(self.train_size + self.test_size + self.valid_size,
                    -1, self.mmGen.frame_size, self.mmGen.frame_size)

        if shuffle:
            np.random.shuffle(data)

        train_data = data[:self.train_size]
        test_data = data[self.train_size:(self.train_size + self.test_size)]
        valid_data = data[(self.train_size + self.test_size):]

        def save_data(path, data):
            if data.shape[0] == 0:
                return

            #n_lead_digits = int(np.log(data.shape[0]) / np.log(10)) + 1
            if Path(path).exists():
                shutil.rmtree(path)
                print("[Warning] Files were removed in ", path)
            os.makedirs(path)

            for i in range(data.shape[0]):
                #f_name = path + str(i + 1).zfill(n_lead_digits) + '.npy'
                f_name = path + str(i) + '.npy'
                np.save(file=f_name, arr=data[i])

        save_data(self.train_path, train_data)
        save_data(self.test_path, test_data)
        save_data(self.valid_path, valid_data)

    def load_data(self, f_names=[], path="", dtype=None):
        """
        Load images from files with pathes in f_names or
        all .npy files from path
        """

        if len(f_names) == 0:
            f_names = [
                path + f for f in os.listdir(path) if f.endswith(".npy")
            ]
        else:
            # Get just existing files
            f_names_tmp = f_names.copy()
            for f in f_names:
                if not os.path.isfile(f):
                    f_names_tmp.remove(f)
            f_names = f_names_tmp

        width, height = self.mmGen.frame_shape
        data = np.empty(
            (len(f_names), self.history_size + self.y_length, width, height),
            dtype=dtype)

        for f, i in zip(f_names, range(0, len(f_names))):
            data[i, :, :, :] = np.load(f)
        return data

    def get_batch_number(self, batch_size, which="train"):
        # Calculates number of available batches
        if which.lower() == "train":
            size_tmp = self.train_size
        elif which.lower() == "test":
            size_tmp = self.test_size
        else:
            size_tmp = self.valid_size

        return int(np.ceil(size_tmp / batch_size))

    def get_batch(self, path, batch_idx, batch_size):
        # Load files from path for corresponding batch_idx
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        f_names = [path + str(i) + '.npy' for i in range(start_idx, end_idx)]
        data_batch = self.load_data(f_names=f_names)

        return data_batch

    def split_x_y(self,
                  data,
                  tensor=True,
                  dtype=None,
                  device="cpu",
                  is_add_channel=True):

        if is_add_channel:
            # Add channel to generalize neural network
            shape = list(data.shape)
            shape.insert(-2, 1)
            data = data.reshape(shape)

        x, y = data[:, :-self.y_length], data[:, -self.y_length:]
        if tensor:
            import torch
            dtype = torch.nn.modules.utils._pair(dtype)

            x = torch.tensor(x, dtype=dtype[0], requires_grad=False).to(device)
            #y = x
            y = torch.tensor(y, dtype=dtype[1], requires_grad=False).to(device)

        return x, y
