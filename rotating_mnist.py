import os
import numpy as np
import torch
from torchvision.datasets.utils import download_url
from PIL import Image, ImageOps
from scipy import ndimage
import idx2numpy
import gzip


class RotatingMnist(object):
    def __init__(
            self,
            root,
            download=False,
            n_t=10**2,
            n_samples=10**2,
            n_same_initial=1,
            initial_random_rotation=True,
            n_angles=2,
            min_angle=-45,  #negative angle => clockwise
            max_angle=45,  #negative angle => counter-clockwise
            angle_std=0.1,  #to pertube an angle
            frame_size=128,
            norm_mean=0.1307,
            norm_std=0.3801,
            specific_digit=None,
            n_styles=np.infty,
            mnist_data_path="./mnist-images-idx3-ubyte.gz",
            mnist_labels_path="./mnist-labels-idx1-ubyte.gz",
            device=torch.device("cpu"),
            name="RotatingMnist",
            model_type='me',
    ):
        self.root = root
        if specific_digit is None:
            digit_name = "digit_random"
        else:
            if not isinstance(specific_digit, list):
                specific_digit = [specific_digit]
            digit_name = "digit_%s" % specific_digit

        self.data_name = "%s_%s_%dx%d_%d_samples_with_%d_timesteps_%d_angles_from_%dd_to_%d.pt" % (
            name, digit_name, frame_size, frame_size, n_samples, n_t, n_angles,
            min_angle, max_angle)
        self.data_file = os.path.join(self.data_folder, self.data_name)
        self.device = device

        # Data-set properties
        self.n_t = n_t
        self.n_samples = n_samples
        self.n_same_initial = n_same_initial  # n. of trajectories from same z0
        self.initial_random_rotation = initial_random_rotation
        self.n_angles = n_angles  # n. of unique angles
        assert min_angle <= max_angle, "min_angle should be <= max_angle"
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle_std = angle_std
        self.model_type = model_type

        # Image properties
        self.mnist = self.load_mnist(mnist_data_path, labels=False)

        # self.labels = idx2numpy.convert_from_file(
        #     gzip.open(mnist_labels_path, 'r'))
        self.labels = self.load_mnist(mnist_data_path, labels=True)

        self.specific_digit = specific_digit
        self.n_styles = n_styles
        self.frame_size = frame_size  # frame size
        self.frame_shape = (self.frame_size, self.frame_size)  # frame size
        self.scale = True
        self.normalize = False
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.data_min = 0
        self.data_max = 255

        print("Path to data: %s" % self.data_file)
        if not self._check_exists():
            print('Dataset not found.' +
                  'If download=False, it will be generated.')
            if download:
                raise NotImplementedError
                self._download()
            else:
                print("Generating data")
                self._generate_dataset()
        else:
            print("Loading existing data")

        self.data = torch.Tensor(torch.load(self.data_file)).to(device)
        # self.data, self.data_min, self.data_max = utils.normalize_data(
        #     self.data)

        t = np.arange(0, n_t) / (n_t - 1)
        self.t = torch.tensor(t).to(self.data)

        self.device = device

    def _generate_dataset(self):
        print('Generating dataset...')
        os.makedirs(self.data_folder, exist_ok=True)
        train_data, labels = self._generate_random_trajectories()
        torch.save(train_data, self.data_file)

        np.savetxt(os.path.splitext(self.data_file)[0] + "_labels.csv",
                   labels.astype(int),
                   fmt='%i',
                   delimiter=',')

    def _download(self):
        print("Downloading the dataset ...")
        os.makedirs(self.data_folder, exist_ok=True)
        url = "????"
        download_url(url, self.data_folder, self.data_name, None)

    def _generate_random_trajectories(self):
        labels = np.zeros((self.n_samples, 2))  # for following inference

        #Same inital we simulate from same rotating point
        mnist = self.mnist  # original data set of mnist
        data = np.zeros(
            (self.n_samples, self.n_t, 1) + self.frame_shape)  #1 is channel

        random_state = np.random.get_state()
        np.random.seed(123)

        if self.specific_digit is not None:
            # Get specific digit indicies
            #sd_ind = np.argwhere(self.labels == self.specific_digit).squeeze()
            #sd_ind = sd_ind[:min(len(labels), self.n_styles)]
            sd_ind = []
            for dig in self.specific_digit:
                sd_ind_ = np.argwhere(self.labels == dig).squeeze()
                sd_ind.extend(sd_ind_[:min(len(labels), self.n_styles)])

        angles = np.linspace(self.min_angle, self.max_angle,
                             self.n_angles)  #me
        # if self.model_type == "me":
        #     angles = np.linspace(self.min_angle, self.max_angle, self.n_angles)
        # elif self.model_type == "examplar":
        #     angles = {}
        #     for dig in self.specific_digit:
        #         angles[dig] = np.random.randint(self.min_angle, self.max_angle,
        #                                         1)[0]
        #     # if len(self.specific_digit) == 2:
        #     #     angles[1] = -angles[0]

        # else:
        #     raise ValueError("Unknown model type for data: " % self.model_type)

        z0_iter = -1  #counter of labels in labels file
        for i in range(self.n_samples):
            print("%04d/%04d" % (i, self.n_samples), end='\r')
            ## Part of code to generate one trajectory
            ## Selecting initial image and position
            if i % self.n_same_initial == 0:
                z0_iter += 1
                if self.specific_digit is not None:
                    chose_digit = np.random.choice(sd_ind, 1)
                else:
                    chose_digit = np.random.choice(mnist.shape[0], 1)
                # Choose from original mnist images and scale to self.frame_size
                mnist_images = Image.fromarray(
                    self.get_picture_array(
                        mnist, chose_digit, shift=0)).resize(
                            (self.frame_size, self.frame_size),
                            Image.ANTIALIAS)

                # Randomly rotate initial point
                digit_z0 = self.arr_from_img(mnist_images)[0]
                if self.initial_random_rotation:
                    angle = np.random.randint(-90, 90, 1)[0]
                    digit_z0 = ndimage.rotate(digit_z0, angle, reshape=False)

            ## Continue to generate the rest of the trajectory
            # Generate angle
            if self.model_type == "me":
                # as ME model for every individual
                angle_id = i % len(angles)
                angle = angles[angle_id]
                labels[i] = (z0_iter, angle_id)
            elif self.model_type == "examplar":
                angle_id = self.labels[chose_digit][0]
                angle = angles[angle_id] + np.random.normal(0, self.angle_std)
                labels[i] = (z0_iter, angle_id)

            # Make one sample of a whole trajectory
            sample = np.empty((self.n_t, 1) + self.frame_shape,
                              dtype=np.float32)  #1 is channel
            digit = digit_z0.copy()
            sample[0, 0, :, :] = np.squeeze(digit)  #initial position
            for frame_idx in range(1, self.n_t):
                digit = ndimage.rotate(digit, angle, reshape=False, cval=0)
                sample[frame_idx, 0, :, :] = np.squeeze(digit)

            sample = np.clip(sample, self.data_min, self.data_max)
            if self.scale:
                sample = (sample - self.data_min) / \
                    (self.data_max - self.data_min)
            if self.normalize:
                sample = (sample - self.norm_mean) / self.norm_std

            data[i] = sample

        # Restore RNG.
        print('%-10s' % 'Done!')
        np.random.set_state(random_state)
        return data, labels

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

        traj = (traj * (self.data_max - self.data_min) +\
          self.data_min)

        def save_image(data, filename):
            im = Image.fromarray(
                np.transpose(data))  #.transpose(Image.ROTATE_90)
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
            width = 1
            gap = 0
            img_h, img_w = img.shape[-2:]

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
            RGB[img != color_placement] = np.stack(
                [img[img != color_placement]] * 3, axis=1)
            # Make frame different color
            RGB[img == color_placement] = [131, 156, 255]

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

    def load_mnist(self, mnist_data_path=None, labels=False):
        # Loads mnist from web if cannot find locally
        from urllib.request import urlretrieve
        import gzip

        if labels:
            file_name = "train-labels-idx1-ubyte.gz"
        else:
            file_name = "train-images-idx3-ubyte.gz"

        if mnist_data_path is None:
            file_path = "%s/%s" % (self.root, file_name)
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
        if labels:
            data = idx2numpy.convert_from_file(gzip.open(file_path, 'r'))
        else:
            with gzip.open(file_path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            #data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
            data = data.reshape(-1, 1, 28, 28)
        return data  #/ np.float32(255)

    def arr_from_img(self, im, shift=0):
        w, h = im.size
        arr = im.getdata()
        #c = int(np.product(arr.size) / (w * h))
        c = len(im.getbands())
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
        s = ('data_file = {data_file}', 'n_samples={n_samples}', 'n_t = {n_t}')
        return '\n'.join(s).format(**self.__dict__)
