import numpy as np
from PIL import Image
from pathlib import Path
import sys
import os
import math


class MovingMnistGenerator(object):
    def __init__(self,
                 out_path,
                 n_samples=1,
                 sample_size=20,
                 digit_size=50,
                 frame_size=200,
                 speed=10,
                 digits_p_img=1,
                 mnist_data_path=None,
                 scale=True,
                 normalize=True,
                 norm_mean=0.1307,
                 norm_std=0.3801):

        self.out_path = out_path + "/"  # path to save results
        if not Path(self.out_path).exists():
            os.makedirs(self.out_path)

        self.n_samples = n_samples
        self.sample_size = sample_size
        self.generated_files = []

        # Image properties
        self.mnist = self.load_mnist(mnist_data_path)
        self.digit_size = digit_size  # MNIST digit size
        self.frame_size = frame_size  # frame size
        self.frame_shape = (self.frame_size, self.frame_size)  # frame size
        self.speed = speed  # moving rate by pixel
        self.digits_p_img = digits_p_img  # number of MNIST digits per frame
        self.normalize = normalize
        self.scale = scale
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def generate(self):
        # Generates and saves set of moving mnist frames
        for seq_idx in range(self.n_samples):
            f_name = self.out_path + str(seq_idx) + '.npy'
            if Path(f_name).exists():
                print("File exists and would be overwritten:", f_name)

            self.generated_files += [f_name]
            samp = self.generate_moving_mnist_sample()
            np.save(file=f_name, arr=samp)

    def load_data(self, f_names=[], path=""):
        # Load images based on file names, default: generated_files
        if len(f_names) == 0:
            f_names = self.generated_files
        if len(f_names) == 0:  # i.e. generated files are also empty
            f_names = [
                path + f for f in os.listdir(path) if f.endswith(".npy")
            ]

        width, height = self.frame_shape
        data = np.empty((len(f_names), self.sample_size, width, height),
                        dtype=np.float32)
        for f, i in zip(f_names, range(0, len(f_names))):
            data[i, :, :, :] = np.load(f)
        return data

    # Below is a block of functions, which generate moving mnist
    # following mostly taken from Tencia Lee via GitHub, from
    # [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations
    # Using LSTMs
    # Srivastava et al

    def generate_moving_mnist_sample(self):
        # Generates and returns video frames in uint8 array
        mnist = self.mnist
        width, height = self.frame_shape
        lims = (x_lim, y_lim) = width - \
            self.digit_size, height - self.digit_size

        sample = np.empty((self.sample_size, width, height), dtype=np.float32)
        # randomly generate direc/speed/position, calculate velocity vector
        direcs = np.pi * (np.random.rand(self.digits_p_img) * 2 - 1)
        # speeds = np.random.randint(5, size=self.digits_p_img)+2
        speeds = np.array(self.speed * np.ones(self.digits_p_img))
        veloc = [(v * math.cos(d), v * math.sin(d))
                 for d, v in zip(direcs, speeds)]
        mnist_images = [
            Image.fromarray(self.get_picture_array(mnist, r, shift=0)).resize(
                (self.digit_size, self.digit_size), Image.ANTIALIAS)
            for r in np.random.randint(0, mnist.shape[0], self.digits_p_img)
        ]
        positions = [(np.random.rand() * x_lim, np.random.rand() * y_lim)
                     for _ in range(self.digits_p_img)]
        for frame_idx in range(self.sample_size):
            canvases = [
                Image.new('L', (width, height))
                for _ in range(self.digits_p_img)
            ]
            canvas = np.zeros((1, width, height), dtype=np.float32)
            for i, canv in enumerate(canvases):
                # map(lambda p: int(round(p)), positions[i]))
                coooords = tuple([int(round(p)) for p in positions[i]])
                canv.paste(mnist_images[i], coooords)
                canvas += self.arr_from_img(canv, shift=0)
            # update positions based on velocity
            next_pos = [(p[0] + v[0], p[1] + v[1])
                        for p, v in zip(positions, veloc)]
            # bounce off wall if a we hit one
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        veloc[i] = tuple(
                            list(veloc[i][:j]) + [-1 * veloc[i][j]] +
                            list(veloc[i][j + 1:]))
            positions = [(p[0] + v[0], p[1] + v[1])
                         for p, v in zip(positions, veloc)]

            # copy additive canvas to data array
            sample[frame_idx, :, :] = np.squeeze(canvas)

        if self.scale:
            sample = (sample - sample.min()) / (sample.max() - sample.min())
        if self.normalize:
            sample = (sample - self.norm_mean) / self.norm_std
        return sample

    def load_mnist(self, mnist_data_path=None):
        # Loads mnist from web if cannot find locally
        from urllib.request import urlretrieve
        import gzip
        file_name = "train-images-idx3-ubyte.gz"
        if mnist_data_path is None:
            file_path = self.out_path + file_name
        else:
            file_path = mnist_data_path

        # Download file if cannot find locally
        if not os.path.exists(file_path):
            print("Downloading %s" % file_name)
            source = 'http://yann.lecun.com/exdb/mnist/'
            urlretrieve(source + file_name, file_path)
        else:
            print("MNIST dataset is loaded from: %s" % file_path)

        # Extract data from file and get right shape
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data  #/ np.float32(255)

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
