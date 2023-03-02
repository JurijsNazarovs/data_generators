import os
import numpy as np
import torch
from torchvision.datasets.utils import download_url
from PIL import Image, ImageOps
from scipy import ndimage
import idx2numpy
import gzip
import math
from sklearn import random_projection
import pandas as pd


class PIB(object):

    def __init__(self,
                 root,
                 load_path,
                 device=torch.device("cpu"),
                 name="PIB",
                 t_max=None,
                 n_dim=1,
                 n_t=3,
                 t_method=''):
        '''
        n_dim: how many dimension to use out of 229 of PET data.
        n_t: number of time_steps to cut. If None, then cut None
        t_method: how to deal with time: window - reshape and ignore t
                  cut - chop first n_t
        '''
        self.n_pac_params = 20  # number of parameters of PAC, excluding id
        self.root = root
        self.load_path = load_path
        self.n_dim = n_dim
        self.max_n_t = 0
        self.n_t = n_t
        self.t_method = t_method
        data_name = load_path.split('/')
        data_name[-1] = data_name[-1].split('.')[0]

        if '' in data_name:
            data_name.remove('')
        data_name = name + '-'.join(data_name)
        if n_t is not None:
            data_name += "-%d_nt" % n_t

        if t_method.strip() != '':
            data_name += "-%s_tmethod" % t_method

        self.data_name = "%s.pt" % data_name
        self.data_file = os.path.join(self.data_folder, self.data_name)
        self.device = device

        print("Path to data: %s" % self.data_file)
        if not self._check_exists():
            print('Dataset not found. It will be generated.')
            print("Generating data")
            self._generate_dataset()
        else:
            print("Loading existing data")

        # Data-set properties
        #self.data = torch.Tensor(torch.load(self.data_file)).to(device)
        self.data = torch.load(self.data_file)
        if not self.n_dim is None:
            # To select only several dimensions
            for i in range(len(self.data)):
                self.data[i][-1] = self.data[i][-1][:, :self.n_dim]

        #self.t = torch.tensor(t).to(self.data)
        for i in self.data:
            self.max_n_t = max(self.max_n_t, len(i[1]))

        if t_max is None:
            all_t = np.concatenate(
                [self.data[i][1] for i in range(len(self.data))])
            self.t_max = np.max(all_t)
        else:
            self.t_max = t_max
        assert self.t_max != 0, "Maximum value of t cannot be 0"
        self.device = device

        # Set labels:
        labels = np.zeros((len(self.data), 1))

        # for i in range(len(labels)):
        #     #take first step and first covariate for each patient as label
        #     labels[i, 0] = int(self.data[i][2][0, 0])
        np.savetxt(os.path.splitext(self.data_file)[0] + "_labels.csv",
                   labels.astype(int),
                   fmt='%i',
                   delimiter=',')

    def _generate_dataset(self):
        print('Generating dataset...')
        os.makedirs(self.data_folder, exist_ok=True)
        train_data = self.collect_data()

        torch.save(train_data, self.data_file)

        np.savetxt(os.path.splitext(self.data_file)[0] + "_column_names.csv",
                   self.column_names,
                   fmt='%s',
                   delimiter='\n')
        # Derive true id based on filtererd id
        ids_true = np.zeros(len(train_data))
        for i in range(len(train_data)):
            curr_id = train_data[i][0][0]
            ind = self.id_true['id'] == curr_id
            ids_true[i] = self.id_true['id_true'].loc[ind]
            
        np.savetxt(os.path.splitext(self.data_file)[0] + "_true_id.csv",
                   ids_true,
                   fmt='%s',
                   delimiter='\n')

    def collect_data(self):
        try:
            df = pd.read_csv(self.load_path, header=0)
            df_wide = df.iloc[:, 4:]._get_numeric_data()
            # Drop columns with age, not many values are available
            df_wide = df_wide[df_wide.columns.drop(
                list(df_wide.filter(regex='age|file|secondary_key')))]
            columns = [i.split('_') for i in df_wide.columns]
            columns_pattern = list(set(["_".join(i[1:]) for i in columns]))
            columns = ["_".join(i[1:]) + "_%s" % i[0] for i in columns]
            df_wide.columns = columns
            df_wide['id'] = df_wide.index
            df_wide["id_true"] = df['reggieid']  # df_wide.index

            df_long = pd.wide_to_long(
                df_wide,
                stubnames=columns_pattern,
                i='id',
                j='time',
                sep="_",
            ).reset_index()

            data = df_long
            data.dropna(axis=0,
                        how='any',
                        thresh=int(0.95 * len(columns_pattern)),
                        subset=columns_pattern,
                        inplace=True)

            #exclude columns with na
            self.id_true = pd.concat([data['id'], data['id_true']], axis=1)
            self.id_true = self.id_true.drop_duplicates(keep='first')

            na_columns = data.columns[data.isna().any()].tolist()
            na_columns.append('id_true')  #remove id_true anyway
            data = data.drop(na_columns, axis=1)

            # Clean up time steps that each group start from 0
            data.sort_values(by=['id', 'time'], inplace=True)
            # data['time'] = data['time'].sub(
            #     data.groupby('id')['time'].transform('first'))
            data['time'] = data.groupby('id').cumcount()

            # Remove data with < n_t steps and decrease the rest to n_t steps
            if self.n_t is not None and self.t_method == 'cut':
                ids_n_t = data.groupby('id')['time'].transform(
                    'last') >= self.n_t - 1
                data = data[ids_n_t].groupby('id').head(self.n_t)

            print('done')
        except:
            raise ValueError("Problems with loading data from %s" %
                             self.load_path)

        # transfer to numpy for easy work
        uniq_ids = sorted(np.unique(data.id))
        patients = []

        #data = data.to_numpy()
        nsteps = data.groupby('id').size()

        print("Number of steps available:\n",
              nsteps.value_counts().sort_values())
        for patient_id in uniq_ids:
            # Skip patient with 1 entree
            ind = data.id == patient_id
            if sum(ind) < 2:
                continue
            data_ = data.loc[ind]
            #ts = data_.time.to_numpy().astype(np.float64)
            ts = self.round(data_.time.to_numpy())
            ids = data_.id.to_numpy()
            data_ = data_.iloc[:, 2:]  #ignore id, time

            # Create rolling window

            if self.n_t is not None and len(
                    ts) > self.n_t and self.t_method == 'window':
                #data_= data_.unfold(dimension=0, size=self.n_t, step=1)
                # hui = np.lib.stride_tricks.sliding_window_view(data_, self.n_t)
                # pi = hui.transpose(0,2,1).reshape(-1, hui.shape[1])
                # print("HI")
                for i in range(len(ts) - self.n_t + 1):
                    ts_window = ts[:self.n_t]  #ts[i: i+self.n_t]
                    data_window = data_[i:i + self.n_t].to_numpy()
                    patients.append([ids, ts_window, data_window])
            else:
                patients.append([ids, ts, data_.to_numpy()])

        self.column_names = data_.columns.to_list()
        print('Done with PIB data')

        return patients

    def visualize(
            self,
            traj,
            concatenate=True,
            plot_path=None,  #'plots/rotmnist/traj',
            img_w=None,
            img_h=None,
            save_separate=False,
            add_frame=True,  #False,
            **kwargs):
        raise NotImplemented("Visualization is not implemented")

    def split_train_test(self, fraction):
        n = len(self.data)  #take length of id
        train_n = int(np.ceil(n * fraction))

        # Maybe need to add shuffling before split
        data_train = self.data[:train_n]
        data_test = self.data[train_n:]

        if len(data_train) == 0 or len(data_test) == 0:
            warnmsg("Warning: length of training or validation data is 0. "
                    "Check --train_perc.")
            if len(data_test) == 0:
                warnmsg("Warning: length of validation data is 0. "
                        "Assigning all data from the training")
                data_test = data_train
            else:
                raise ValueError("Length of training data is 0. "
                                 "Consider to increase --train_perc.")

        return data_train, data_test

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

    # @classmethod
    def variable_time_collate_fn(self,
                                 batch,
                                 args,
                                 device=torch.device("cpu"),
                                 data_type="train"):
        """
        Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
        - record_id is a patient id
        - tt is a 1-dimensional tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        - labels is a list of labels for the current patient, if labels are available. Otherwise None.
        Returns:
        - combined_tt: The union of all time observations.
        - combined_vals: (M, T, D) tensor containing the observed values.
        - combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
        """

        # Batch = [id, time, x (independent on time), y (dependent on time)]

        # At the end I want to return the following:
        # combined time series, where for each time we have corresponding
        # values of y (PET data) to predict
        # and values of x (PAC) data which are the same for time steps and
        # do not require time steps.
        # Thus, I will have combined t, combined y, corresponding mask and separate
        # x values (PAC), which are independent of time.

        d_y = batch[0][-1].shape[
            1]  # dimension of y (PET) data to combine (time)

        #combined_tt = torch.cat([torch.tensor(ex[1]) for ex in batch])
        combined_tt = np.concatenate([ex[1] for ex in batch])
        #combined_tt = np.arange(self.max_n_t)
        combined_tt = torch.tensor(np.unique(self.round(combined_tt)))
        #torch.cat([torch.tensor(ex[1]) for ex in batch]),
        combined_tt, inverse_indices = torch.unique(combined_tt,
                                                    sorted=True,
                                                    return_inverse=True)

        # if torch.max(combined_tt) != 0.:
        #     combined_tt = combined_tt / torch.max(combined_tt)
        #combined_tt = combined_tt / self.t_max
        combined_tt = combined_tt.to(device)

        #offset = 0
        combined_vals_y = torch.zeros([len(batch),
                                       len(combined_tt), d_y]).to(device)
        combined_mask_y = torch.zeros([len(batch),
                                       len(combined_tt), d_y]).to(device)

        #combined_labels = None
        #N_labels = 1

        #combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(
        #    float('nan'))
        #combined_labels = combined_labels.to(device=device)

        # Here vals_x are values of PAC and vals_y are values of PET
        #for b, (record_id, tt, vals_x, vals_y) in enumerate(batch):
        for b, (record_id, tt, vals_y) in enumerate(batch):
            #tt = tt  #.to(device)

            #indices = inverse_indices[offset:offset + len(tt)]
            #offset += len(tt)
            indices = np.where(
                np.in1d(combined_tt.data.cpu().numpy(), self.round(tt)))[0]

            combined_vals_y[b, indices] = torch.tensor(
                vals_y,
                dtype=combined_vals_y.dtype,
                device=combined_vals_y.device)
            combined_mask_y[b, indices] = 1.

        # Normalizing through (0, 1) - batch, time
        combined_vals_y = (combined_vals_y - combined_vals_y.mean(
            (0, 1))) / combined_vals_y.std((0, 1))
        combined_vals_y[~combined_mask_y.to(bool)] = 0

        data_dict = {
            "data": combined_vals_y,
            "mask": combined_mask_y,
            "time_steps": combined_tt / self.t_max,
            #"condition": combined_vals_x,
        }

        # data_dict = data_utils.split_and_subsample_batch(data_dict,
        #                                                  args,
        #                                                  data_type=data_type)
        return data_dict

    def round(self, x):
        # Necessary to preserve the same rounding,
        # to use np.unique with float point
        return x.round(2)


def warnmsg(msg):
    print("******************************")
    print("Warning:")
    print(msg)
    print("******************************")
