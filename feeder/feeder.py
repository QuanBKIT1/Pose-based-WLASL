from . import tools
import numpy as np
import pickle
from torch.utils.data import Dataset
import random

flip_index = {'27-sam-v1': np.concatenate(([0, 2, 1, 4, 3, 6, 5], [17, 18, 19, 20, 21, 22, 23, 24, 25, 26], [
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), axis=0),
    '27-sam': np.array([0, 2, 1, 4, 3, 6, 5, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
    '31-our': np.array([0, 2, 1, 4, 3, 6, 5, 9, 8, 7, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
    '49-our': np.array([0, 2, 1, 4, 3, 6, 5, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                        44, 45, 46, 47, 48, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),
    '53-our': np.array([0, 2, 1, 4, 3, 6, 5, 9, 8, 7, 10, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])}


class Feeder(Dataset):
    def __init__(self, data_path, label_path, max_xy=256,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, random_mirror=False, random_mirror_p=0.5, is_vector=False):
        """

        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.max_xy = max_xy
        self.load_data()
        self.is_vector = is_vector
        if normalization:
            self.get_mean_map()
        print(f"""
        Setting with:
              random_choose: {random_choose},
              random_shift: {random_shift} 
              random_move: {random_move} 
              window_size: {window_size} 
              normalization: {normalization} 
              max_width_height: {max_xy}
              Joint mode: {self.data.shape[3]}
        Load {len(self.label)} samples in {self.data_path} 
""")

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(
                    f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(
                self.data_path, mmap_mode='r').astype(np.float32)
        else:
            self.data = np.load(self.data_path).astype(np.float32)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(
            axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)

        if self.random_mirror:
            if random.random() > self.random_mirror_p:
                if data_numpy.shape[2] == 27 and self.max_xy == 512:
                    data_numpy = data_numpy[:, :, flip_index['27-sam-v1'], :]
                elif data_numpy.shape[2] == 27 and self.max_xy == 256:
                    data_numpy = data_numpy[:, :, flip_index['27-sam'], :]
                elif data_numpy.shape[2] == 53 and self.max_xy == 256:
                    data_numpy = data_numpy[:, :, flip_index['53-our'], :]
                elif data_numpy.shape[2] == 31 and self.max_xy == 256:
                    data_numpy = data_numpy[:, :, flip_index['31-our'], :]
                elif data_numpy.shape[2] == 49 and self.max_xy == 256:
                    data_numpy = data_numpy[:, :, flip_index['49-our'], :]
                
                if self.is_vector:
                    data_numpy[0, :, :, :] = - data_numpy[0, :, :, :]
                else:
                    # data_numpy[0, :, :, :] = 512 - data_numpy[0, :, :, :]
                    data_numpy[0, :, :, :] = self.max_xy - \
                        data_numpy[0, :, :, :]

        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            assert data_numpy.shape[0] == 3
            if self.is_vector:
                data_numpy[0, :, 0, :] = data_numpy[0, :, 0, :] - \
                    data_numpy[0, :, 0, 0].mean(axis=0)
                data_numpy[1, :, 0, :] = data_numpy[1, :, 0, :] - \
                    data_numpy[1, :, 0, 0].mean(axis=0)
            else:
                data_numpy[0, :, :, :] = data_numpy[0, :, :, :] - \
                    data_numpy[0, :, 0, 0].mean(axis=0)
                data_numpy[1, :, :, :] = data_numpy[1, :, :, :] - \
                    data_numpy[1, :, 0, 0].mean(axis=0)

        if self.random_shift:
            if self.max_xy == 256:
                cor_shift = 10
            else:
                cor_shift = 20
            if self.is_vector:
                data_numpy[0, :, 0, :] += random.random() * \
                    cor_shift * 2 - cor_shift
                data_numpy[1, :, 0, :] += random.random() * \
                    cor_shift * 2 - cor_shift
            else:
                data_numpy[0, :, :, :] += random.random() * \
                    cor_shift * 2 - cor_shift
                data_numpy[1, :, :, :] += random.random() * \
                    cor_shift * 2 - cor_shift

        # if self.random_shift:
        #     data_numpy = tools.random_shift(data_numpy)

        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
