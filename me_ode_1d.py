import os
import numpy as np
import torch
from lib.utils import get_dict_template
import lib.utils as utils
from torchvision.datasets.utils import download_url


class MEODE1d(object):
    def __init__(self,
                 root,
                 download=False,
                 min_t=0.,
                 max_t=2.,
                 n_t=10,
                 n_samples=10**2,
                 dt=10**-3,
                 y0_mean=1.3,
                 y0_std=0.01,
                 fix_eff=0.3,
                 rand_eff_std=0.1,
                 device=torch.device("cpu"),
                 name="ME_ODE_1d"):
        # Record information for a print function
        self.root = root
        self.data_name = "%s_%d_samples_with_%d_timesteps_in_[%.2f_%.2f].pt" % (
            name, n_samples, n_t, min_t, max_t)
        self.data_file = os.path.join(self.data_folder, self.data_name)
        self.device = device

        self.min_t = min_t
        self.max_t = max_t
        #self.dt=dt
        self.n_t = n_t
        self.n_samples = n_samples

        self.y0_mean = y0_mean
        self.y0_std = y0_std
        self.fix_eff = fix_eff
        self.rand_eff_std = rand_eff_std

        # Save random seed and restore later
        t = np.arange(min_t, max_t, dt)
        random_state = np.random.get_state()
        np.random.seed(123)
        obs_t = np.sort(np.random.choice(range(len(t)), n_t, replace=False))
        np.random.set_state(random_state)

        if not self._check_exists():
            print('Dataset not found.' +
                  'If download=False, it will be generated.')
            if download:
                raise NotImplementedError
                self._download()
            else:
                print("Generating data")
                self._generate_dataset(t, obs_t)
        else:
            print("Loading existing data")

        self.data = torch.Tensor(torch.load(self.data_file)).to(device)
        self.t = torch.tensor(t[obs_t]).to(self.data)

        # self.data, self.data_min, self.data_max =\
        #     utils.normalize_data(self.data)

    def _download(self):
        print("Downloading the dataset ...")
        os.makedirs(self.data_folder, exist_ok=True)
        url = "????"
        download_url(url, self.data_folder, self.data_name, None)

    def _generate_dataset(self, t, obs_t):
        # t is vector of time points for which to generate ME ODE
        print('Generating dataset...')
        os.makedirs(self.data_folder, exist_ok=True)
        data = self._generate_random_trajectories(t)
        data = data[:, obs_t].reshape(-1, len(obs_t), 1)

        torch.save(data, self.data_file)

    def _generate_random_trajectories(self, t):
        rand_eff = lambda: np.random.normal(
            loc=0, scale=self.rand_eff_std, size=(self.n_samples, 1))
        rand_eff_samples = rand_eff()

        data = make_sde_data(
            t,
            y0_mean=self.y0_mean,
            y0_std=self.y0_std,
            n_sim=self.n_samples,
            a=lambda x: x * (self.fix_eff + rand_eff_samples),
            b=lambda x: 0,  #we make ode data
            b_prime=lambda x: 0)  #.transpose(1, 0)
        #data = torch.tensor(data).float().to(device)

        return data

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
             'min_t = {min_t}', 'max_t = {max_t}', 'y0_mean = {y0_mean}',
             'y0_std = {y0_std}', 'fix_eff = {fix_eff}',
             'rand_eff_std = {rand_eff_std}')
        return '\n'.join(s).format(**self.__dict__)

    def visualize(self, data):
        # Data: batch_size, time_steps
        self.fig = plt.figure(figsize=(10, 4), facecolor='white')
        self.ax = self.fig.add_subplot(111, frameon=False)
        plt.show(block=False)


def make_sde_data(ts,
                  y0_mean=0,
                  y0_std=10**-3,
                  n_sim=1,
                  a=None,
                  b=None,
                  b_prime=None):
    # Milstein Method to generate data for SDE:
    # dX_t = a(X_t)dt + b(X_t)dW_t
    # Generate as (D means delta, e.g. Dt is delta t):
    # Y_n+1 = Y_n + a(Y_n)Dt + b(Y_n)DW_n + 1/2b(Y_n)b'(Y_n)((DW_n)**2 - Dt)
    # Input:
    # ts: time points to simulate sde
    # y0: initial value (size?)
    # n_sim: number of simulated process
    # a, b: functions, which returns shape [n_sim; 1]
    # b_prime: derivative
    # Output: generated data ys [n_sim, len(ts)]

    dt = float(ts[1] - ts[0])

    # dw Random process
    def dW(delta_t):
        """" Random sample normal distribution"""
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

    # Simulation
    ys = np.zeros((n_sim, len(ts)))
    # Initial point with randomness for VAE
    ys[:, 0] = y0_mean + np.random.normal(loc=0.0, scale=y0_std, size=n_sim)
    # Simulation of the rest steps
    for j in range(1, len(ts)):
        y = ys[:, j - 1].reshape(-1, 1)
        # Milstein method
        ys[:, j] = (y + a(y) * dt + b(y) * dW(dt) +\
            1/2*b(y)*b_prime(y) * (dW(dt)**2 - dt)).reshape(-1)
    return ys
