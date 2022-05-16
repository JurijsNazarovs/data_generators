import os
import numpy as np
import torch
import lib.utils as utils
from torchvision.datasets.utils import download_url


class HopperPhysics(object):
    def __init__(
            self,
            root,
            download=False,
            n_t=10**2,
            n_samples=10**2,
            n_same_initial=1,
            n_angles=2,
            min_angle=-3,  #negative angle => clockwise
            max_angle=3,  #negative angle => counter-clockwise
            device=torch.device("cpu"),
            steps_to_skip=20,
            min_height=1000,  #to avoid hitting ground
            state=['hopper', 'stand'],
            name="Mujoco"):
        self.root = root
        self.data_name = "%s_%s_%d_samples_with_%d_timesteps_%d_speeds_from_%d_to_%d.pt" % (
            name, "-".join(state), n_samples, n_t, n_angles, min_angle,
            max_angle)
        self.data_file = os.path.join(self.data_folder, self.data_name)
        self.device = device

        self.n_t = n_t
        self.state = state
        self.n_samples = n_samples
        self.n_same_initial = n_same_initial
        self.steps_to_skip = steps_to_skip
        self.n_angles = n_angles  #n. of unique angles
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_height = min_height

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
        self.data, self.data_min, self.data_max = HopperPhysics.normalize_data(
            self.data)
        t = np.arange(0, n_t) / (n_t - 1)
        self.t = torch.tensor(t).to(self.data)
        self.device = device

    def _generate_dataset(self):
        print('Generating dataset ...')
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
            from dm_control import suite
        except ImportError as e:
            raise Exception(
                'Deepmind Control Suite is required to generate the dataset.'
            ) from e
        try:
            env = suite.load(*self.state)
        except:
            raise ValueError(
                "Current state: %s was not found in dm_control suite" %
                self.state)
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
        Dpos = len(physics.data.qpos)
        Dvel = len(physics.data.qvel)

        # Store the state of the RNG to restore later.
        random_state = np.random.get_state()
        np.random.seed(123)
        data = np.zeros(
            (self.n_samples, self.n_t, Dpos + Dvel))  #2: qpos, qvel

        # Generate initial conditions the same for several samples
        n_samples_ = self.n_samples // self.n_same_initial + 1
        initial_qpos = np.zeros((n_samples_, Dpos))
        if 'hopper' in self.state:
            # 0:2 is position(x, z), 2:6 is quaternion position - rotation
            # We want z > 0 for hopper to stay above ground. To change x: [:, [0]]
            initial_qpos[:, [1]] = np.random.uniform(0, 0.5, size=(n_samples_, 1)) +\
                self.min_height # + large number that he never hit the ground
            # We randomly sample rotation matrix (4 coordinates)
            initial_qpos[:, 2:Dpos] = np.random.uniform(-2,
                                                        2,
                                                        size=(n_samples_,
                                                              Dpos - 2))
            self.min_angle, self.max_angle = -5, 5
        elif 'walker' in self.state:
            initial_qpos[:, [0]] = 0  #z
            initial_qpos[:, [1]] = 0  #x
            initial_qpos[:, 2:Dpos] = np.random.uniform(-1,
                                                        1,
                                                        size=(n_samples_,
                                                              Dpos - 2))
            self.min_angle, self.max_angle = -2, 2
        elif 'cheetah' in self.state:
            initial_qpos[:, [0]] = 0  #x
            initial_qpos[:, [1]] = 0  #z
            initial_qpos[:, [2]] = np.random.uniform(-1,
                                                     0,
                                                     size=(n_samples_, 1))  #y
            initial_qpos[:, 3:Dpos] = np.random.uniform(0,
                                                        0.1,
                                                        size=(n_samples_,
                                                              Dpos - 3))
            self.min_angle, self.max_angle = 0, 3
            self.steps_to_skip = 5
        elif 'humanoid' in self.state:
            initial_qpos[:, :
                         3] = 0  #np.random.uniform(0, 1, size=(n_samples_, 3))

            initial_qpos[:, 3:7] = 0
            initial_qpos[:, 7:Dpos] = np.random.uniform(0.1,
                                                        1,
                                                        size=(n_samples_,
                                                              Dpos - 7))
            self.min_angle, self.max_angle = -1, 1
        elif 'cartpole' in self.state:
            # [0] is a position on the bar, rest is position of each joint
            initial_qpos[:, :Dpos] = np.random.uniform(0,
                                                       1,
                                                       size=(n_samples_, Dpos))
            self.min_angle, self.max_angle = -3, 3
            self.steps_to_skip = 20
        elif 'swimmer' in self.state:
            initial_qpos[:, [0]] = 0  #z
            initial_qpos[:, [1]] = 0  #x
            initial_qpos[:, 2:Dpos] = np.random.uniform(-5,
                                                        5,
                                                        size=(n_samples_,
                                                              Dpos - 2))
            self.min_angle, self.max_angle = -4, 4
        elif 'lqr' in self.state:
            initial_qpos[:, [0]] = 0  #z
            initial_qpos[:, [1]] = 0  #x
            initial_qpos[:, 2:Dpos] = np.random.uniform(0,
                                                        0.1,
                                                        size=(n_samples_,
                                                              Dpos - 2))
            self.min_angle, self.max_angle = 0.1, 1

        speeds = np.linspace(self.min_angle, self.max_angle, self.n_angles)

        # Generate random effect and add it to rotation
        for i in range(self.n_samples):
            with physics.reset_context():
                physics.data.qpos[:] = initial_qpos[i // self.n_same_initial]
                physics.data.qvel[:] = 0*np.random.uniform(
                -5, 5, size=physics.data.qvel.shape) + \
                speeds[i % len(speeds)]

            # print(physics.height())
            # print(physics.speed())

            for t in range(self.n_t):
                data[i, t, :Dpos] = physics.data.qpos
                data[i, t, Dpos:] = physics.data.qvel

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
            from dm_control import suite
        except ImportError as e:
            raise Exception(
                'Deepmind Control Suite is required to visualize the dataset.'
            ) from e

        try:
            env = suite.load(*self.state)
        except:
            raise ValueError(
                "Current state: %s was not found in dm_control suite" %
                self.state)
        physics = env.physics
        Dpos = len(physics.data.qpos)
        Dvel = len(physics.data.qvel)

        try:
            T, _ = traj.size()
            traj = traj.cpu() * self.data_max.cpu() + self.data_min.cpu()
        except:
            # In case we gave numpy trajectory with differently defined size()
            T, _ = traj.shape
            traj = traj * self.data_max.cpu().numpy() +\
                self.data_min.cpu().numpy()

        try:
            from PIL import Image  # noqa: F401
        except ImportError as e:
            raise Exception('PIL is required to visualize the dataset.') from e

        def save_image(data, filename):
            im = Image.fromarray(data)
            im.save(filename)

        if plot_path is not None:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        if concatenate:
            concat_image = []

        for t in range(T):
            with physics.reset_context():
                physics.data.qpos[:] = traj[t, :Dpos]
                physics.data.qvel[:] = traj[t, Dpos:]
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
        s = ('data_file = {data_file}', 'n_samples={n_samples}', 'n_t = {n_t}')
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

    @classmethod
    def get_states(obj):
        # Software to list taks, datasets, and number of parameters to simulate
        from dm_control import suite  # noqa: F401
        for task in suite.ALL_TASKS:  # list all tasks and datasets
            env = suite.load(*task)
            physics = env.physics
            print(task, len(physics.data.qpos), len(physics.data.qvel))
