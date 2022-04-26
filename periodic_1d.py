import os
import numpy as np
import torch
import lib.utils as utils
from torchvision.datasets.utils import download_url


class Periodic1d(object):
    def __init__(
            self,
            root,
            download=False,
            min_t=0.,
            max_t=2.,
            n_t=10,
            n_samples=10**2,
            dt=10**-3,
            y0_mean=2.0,
            y0_std=0.01,
            init_freq=0.3,
            init_ampl=1,
            n_models=1,
            model_type='amplitude',
            final_freq=None,
            final_ampl=None,
            device=torch.device("cpu"),
            name="Periodic_1d",
    ):
        # Record information for a print function
        # model_type generate:
        #  - amp: different amplitudes
        #  - freq: different freq
        #  - y0: different initial point z0
        # Can be mix of all, e.g. ampfreqy0
        self.root = root
        self.data_name =\
            "%s_%d_samples_with_%d_timesteps_in_[%.2f_%.2f]_from_%d_sources_diff_%s.pt" % (
                name, n_samples, n_t, min_t, max_t, n_models, model_type)
        self.data_file = os.path.join(self.data_folder, self.data_name)
        self.device = device

        self.min_t = min_t
        self.max_t = max_t
        self.n_t = n_t
        self.n_samples = n_samples
        self.n_models = n_models
        self.model_type = model_type

        self.y0_mean = y0_mean
        self.y0_std = y0_std

        self.init_freq = init_freq
        self.init_ampl = init_ampl
        self.final_freq = final_freq
        self.final_ampl = final_ampl

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

    def _download(self):
        print("Downloading the dataset ...")
        os.makedirs(self.data_folder, exist_ok=True)
        url = "????"
        download_url(url, self.data_folder, self.data_name, None)

    def _generate_dataset(self, t, obs_t):
        # t is vector of time points for which to generate data
        print('Generating dataset...')
        os.makedirs(self.data_folder, exist_ok=True)
        data = self._generate_random_trajectories(t)
        data = data[:, obs_t].reshape(-1, len(obs_t), 1)

        torch.save(data, self.data_file)

    def _generate_random_trajectories(self, t):
        data = make_data(t,
                         y0_mean=self.y0_mean,
                         y0_std=self.y0_std,
                         n_sim=self.n_samples,
                         init_freq=self.init_freq,
                         init_ampl=self.init_ampl,
                         final_freq=self.final_freq,
                         final_ampl=self.final_ampl,
                         n_models=self.n_models,
                         model_type=self.model_type)

        return data

    def _check_exists(self):
        return os.path.exists(self.data_file)

    @property
    def data_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

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
             'y0_std = {y0_std}', 'init_freq = {init_freq}',
             'init_ampl = {init_ampl}', 'final_freq = {final_freq}',
             'final_ampl = {final_ampl}')
        return '\n'.join(s).format(**self.__dict__)

    def visualize(self, data):
        # Data: batch_size, time_steps
        self.fig = plt.figure(figsize=(10, 4), facecolor='white')
        self.ax = self.fig.add_subplot(111, frameon=False)
        plt.show(block=False)


def get_next_val(init, t, tmin, tmax, final=None):
    if final is None:
        return init
    val = init + (final - init) / (tmax - tmin) * t
    return val


def make_data(ts,
              y0_mean=0,
              y0_std=10**-3,
              n_sim=1,
              init_freq=0.3,
              init_ampl=1,
              final_freq=None,
              final_ampl=None,
              phi_offset=0.,
              n_models=2,
              model_type='amplitude'):
    # Input:
    # ts: time points to simulate sde
    # y0: initial value (size?)
    # n_sim: number of simulated process
    # n_models: how many models to generate from the same initial value
    # Output: generated data ys [n_sim, len(ts)]

    # Simulation
    tmin = ts.min()
    tmax = ts.max()

    ys = np.zeros((n_sim, len(ts)))

    # Get coefficients to generate different models
    # and initial point with randomness for VAE
    if n_models == 2:
        amp_coef = np.array((1, -1))
        freq_coef = np.array((1, 0.3))
        y0_mean_ = np.array((2, -2))
    else:
        amp_coef = np.random.rand(n_models) * 2 - 1  #runif(0, 1)
        freq_coef = np.random.rand(n_models) * 2 - 1  #np.array(1)
        y0_mean_ = np.random.normal(loc=y0_mean, scale=2, size=n_models)

    if 'freq' not in model_type:
        freq_coef = 1
    if 'amp' not in model_type:
        amp_coef = 1
    if 'y0' in model_type:
        y0_mean = y0_mean_  # get the one gerated for different groups

    amp_coef = np.resize(amp_coef, n_sim)  # fit it to size of bs
    freq_coef = np.resize(freq_coef, n_sim)  # fit it to size of bs
    y0_mean = np.resize(y0_mean, n_sim)

    # Simulation of the rest steps
    ys[:, 0] = y0_mean + np.random.normal(loc=0.0, scale=y0_std, size=n_sim)
    t_prev = ts[0]
    phi = phi_offset

    for j in range(1, len(ts)):
        t = ts[j]
        dt = t - t_prev
        amp = get_next_val(init_ampl, t, tmin, tmax, final_ampl)
        freq = get_next_val(init_freq, t, tmin, tmax, final_freq)
        phi = phi + 2 * np.pi * freq * dt * freq_coef

        ys[:, j] = amp_coef * amp * np.sin(phi) + ys[:, 0]
        t_prev = t

    return ys
