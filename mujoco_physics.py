import os
import numpy as np
import torch
import lib.utils as utils
from torchvision.datasets.utils import download_url


class HopperPhysics(object):
    def __init__(self,
                 root,
                 download=False,
                 n_t=10**2,
                 n_samples=10**2,
                 n_same_initial=1,
                 n_angles=2,
                 fix_eff=0.3,
                 rand_eff_std=0.1,
                 device=torch.device("cpu"),
                 steps_to_skip=20,
                 name="Hopper"):
        self.root = root
        self.data_name = "%s_%d_samples_with_%d_timesteps_%d_speeds.pt" % (
            name, n_samples, n_t, n_angles)
        self.data_file = os.path.join(self.data_folder, self.data_name)
        self.device = device

        self.n_t = n_t
        self.n_samples = n_samples
        self.n_same_initial = n_same_initial
        self.steps_to_skip = steps_to_skip
        self.n_angles = n_angles  #n. of unique angles

        self.fix_eff = fix_eff
        self.rand_eff_std = rand_eff_std

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
        self.data, self.data_min, self.data_max = HopperPhysics.normalize_data(
            self.data)
        t = np.arange(0, n_t) / (n_t - 1)
        self.t = torch.tensor(t).to(self.data)
        self.device = device

    def _generate_dataset(self):
        print('Generating dataset...')
        os.makedirs(self.data_folder, exist_ok=True)
        train_data = self._generate_random_trajectories()
        torch.save(train_data, self.data_file)

    def _download(self):
        print("Downloading the dataset ...")
        os.makedirs(self.data_folder, exist_ok=True)
        url = "????"
        download_url(url, self.data_folder, self.data_name, None)

    def _generate_random_trajectories(self):

        try:
            from dm_control import suite  # noqa: F401
            env = suite.load('hopper', 'stand')
        except ImportError as e:
            raise Exception(
                'Deepmind Control Suite is required to generate the dataset.'
            ) from e

        # try:
        #     #from gym.envs.mujoco import mujoco_env
        #     #env = suite.load('hopper', 'stand')
        #     import gym
        #     import pybulletgym  #free open source engine for hopper
        #     #env = gym.make('Hopper-v2')
        #     env = gym.make("HopperMuJoCoEnv-v0")  #from pybulletgym
        # except ImportError as e:
        #     raise Exception(
        #         'Deepmind Control Suite is required to generate the dataset.'
        #     ) from e

        physics = env.physics
        D = len(physics.data.qpos)

        # Store the state of the RNG to restore later.
        random_state = np.random.get_state()
        np.random.seed(123)
        data = np.zeros((self.n_samples, self.n_t, 2 * D))  #2: qpos, qvel
        # Generate initial conditions the same for several samples
        n_samples_ = self.n_samples // self.n_same_initial + 1
        # x and z positions of the hopper.
        # We want z > 0 for the hopper to stay above ground.
        # 0:2 is position, 3:6 is quaternion position - rotation
        initial_qpos = np.zeros((n_samples_, D))
        initial_qpos[:, :2] = np.random.uniform(0, 0.5, size=(n_samples_, 2))
        initial_qpos[:, 2:] = np.random.uniform(-2,
                                                2,
                                                size=(n_samples_, D - 2))

        speeds = np.linspace(0.1, 5, self.n_angles)
        # Generate random effect and add it to rotation
        for i in range(self.n_samples):
            with physics.reset_context():
                physics.data.qpos[:] = initial_qpos[i // self.n_same_initial]
                physics.data.qvel[:] = 0*np.random.uniform(
                    -5, 5, size=physics.data.qvel.shape) + \
                    speeds[i % len(speeds)]

            for t in range(self.n_t):
                data[i, t, :D] = physics.data.qpos
                data[i, t, D:] = physics.data.qvel

                for _ in range(self.steps_to_skip):
                    #skip up to N steps, to make it less degenerate
                    physics.step()

        # Restore RNG.
        np.random.set_state(random_state)
        return data

    def visualize(self,
                  traj,
                  concatenate=False,
                  plot_path=None,
                  img_h=80,
                  img_w=80,
                  save_separate=False,
                  **kwargs):
        r"""Generates images of the trajectory and stores them as
            <dirname>/traj<index>-<t>.jpg"""

        try:
            T, D = traj.size()
            traj = traj.cpu() * self.data_max.cpu() + self.data_min.cpu()
        except:
            # In case we gave numpy trajectory with differently defined size()
            T, D = traj.shape
            traj = traj * self.data_max.cpu().numpy() +\
                self.data_min.cpu().numpy()

        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception(
                'Deepmind Control Suite is required to visualize the dataset.'
            ) from e

        try:
            from PIL import Image  # noqa: F401
        except ImportError as e:
            raise Exception('PIL is required to visualize the dataset.') from e

        def save_image(data, filename):
            im = Image.fromarray(data)
            im.save(filename)

        if plot_path is not None:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        env = suite.load('hopper', 'stand')
        physics = env.physics

        if concatenate:
            concat_image = []

        for t in range(T):
            with physics.reset_context():
                physics.data.qpos[:] = traj[t, :D // 2]
                physics.data.qvel[:] = traj[t, D // 2:]
            image = physics.render(height=img_h, width=img_w, camera_id=0)
            if concatenate:
                concat_image.append(image)
            if plot_path is not None and save_separate:
                save_image(image, ('_%03d.' % t).join(plot_path.split('.')))

        if concatenate:
            concat_image = np.concatenate(concat_image, axis=1)
            if plot_path is not None:
                save_image(concat_image, '_concat.'.join(plot_path.split('.')))

            return concat_image

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

    @staticmethod
    def normalize_data(data):
        reshaped = data.reshape(-1, data.size(-1))

        att_min = torch.min(reshaped, 0)[0]
        att_max = torch.max(reshaped, 0)[0]

        # we don't want to divide by zero
        att_max[att_max == 0.] = 1.

        if (att_max != 0.).all():
            data_norm = (data - att_min) / att_max
        else:
            raise Exception("Zero!")

        if torch.isnan(data_norm).any():
            raise Exception("nans!")

        return data_norm, att_min, att_max
