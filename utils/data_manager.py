import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000
from tqdm import tqdm
import pickle
import torch
import random
import scipy.io as sio

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self.Random_Matrix = None
        self.transformed_clean_data = None

        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_accumulate_tasksize(self, task):
        return sum(self._increments[:task + 1])

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
            self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y, snr = self._train_data, self._train_targets, self._train_snr
        elif source == "test":
            x, y, snr = self._test_data, self._test_targets, self._test_snr
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            # //
            # trsf = transforms.Compose([
            #     transforms.ToTensor(),
            #     None,
            # ])
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        # //
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets, snrs = [], [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets, class_snr = self._select(
                    x, y, snr, low_range=idx, high_range=idx + 1)
            else:
                class_data, class_targets, class_snr = self._select_rmm(
                    x, y, snr, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)
            snrs.append(class_snr)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets, appendent_snr = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
            snrs.append(appendent_snr)

        data, targets, snrs = np.concatenate(data), np.concatenate(targets), np.concatenate(snrs)

        # 新增：获取干净信号（同类型且SNR=18）
        # ------------------------------
        if self.transformed_clean_data is None:
            clean_data = []
            # 预构建类别到SNR=18信号的映射（提高查找效率）
            class_clean_signal_map = {}

            for class_idx in np.unique(targets):
                # 筛选该类别下SNR=18的信号
                mask = (y == class_idx) & (np.isclose(snr, 18.0))
                if np.sum(mask) == 0:
                    raise ValueError(f"类别 {class_idx} 没有SNR=18的信号，请检查数据集")
                class_clean_signal_map[class_idx] = x[mask]

            # 为每个样本匹配同类别干净信号
            for i in range(len(targets)):
                class_idx = targets[i]
                # 从该类别干净信号中随机选择一个（可根据需求改为固定选择）
                clean_sig = random.choice(class_clean_signal_map[class_idx])
                clean_data.append(clean_sig)

            clean_data = np.stack(clean_data, axis=0)  # 转换为数组格式
            # 应用数据变换
            self.transformed_clean_data = [trsf(c) for c in clean_data]  # 干净信号也应用相同变换
        if ret_data:
            return data, targets, snrs, DummyDataset(data, targets, snrs, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, snrs, trsf, self.use_path)

    def get_finetune_dataset(self, known_classes, total_classes, source, mode, appendent, type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx + 1)
            num = len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot * (total_classes - known_classes) / known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot / (total_classes - known_classes))
        for idx in range(known_classes, total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            val_indx = np.random.choice(len(class_data), new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data = np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_dataset_with_split(
            self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        with open('/root/autodl-tmp/RML2018_sampled_480k.pkl', 'rb') as f:
            data = pickle.load(f)
        X = data['X']  # 已经是 (480000, 2, 1024) 形状
        lbl = data['lbl']

        # 获取所有的 mod 和 snr 类别（用于后续找索引或映射）
        mods = sorted(list(set([l[0] for l in lbl])))
        snrs = sorted(list(set([l[1] for l in lbl])))
        # 将所有数据沿第一个维度堆叠成三维数组（总样本数, 2, 128）(220*1000个样本)
        n_examples = X.shape[0]
        n_train = int(0.75 * n_examples)
        n_valid = int(0.25 * n_examples)

        # 生成随机打乱的索引列表（可替换为预定义的shuffle文件）
        allnum = list(range(0, n_examples))
        if shuffle:
            # 关键修改：设置随机种子
            random.seed(seed)  # 固定Python随机模块种子
            np.random.seed(seed)  # 固定NumPy随机种子
            random.shuffle(allnum)

        # 划分训练/验证/测试集索引
        train_idx = allnum[0:n_train]
        # valid_idx = allnum[n_train:n_train + n_valid]
        test_idx = allnum[n_train:]

        # 构建训练集（数据+分类标签+SNR标签）
        X_train = X[train_idx]
        Y_train = list(map(lambda x: mods.index(lbl[x][0]), train_idx))  # 调制类型转换为类别索引
        Z_train = list(map(lambda x: lbl[x][1], train_idx))  # 提取SNR值

        # 构建测试集
        X_test = X[test_idx]
        Y_test = list(map(lambda x: mods.index(lbl[x][0]), test_idx))
        Z_test = list(map(lambda x: lbl[x][1], test_idx))

        self._train_data = X_train  # data_train['data']
        self._train_targets = Y_train  # data_train['labels']
        self._train_snr = Z_train
        self._test_data = X_test  # data_test['data']
        self._test_targets = Y_test  # data_test['labels']
        self._test_snr = Z_test

        self.use_path = False  # As we are using raw image data, not paths.

        # Transforms (assuming that you still need them)
        self._train_trsf = []  # Define your transformations here if needed
        self._test_trsf = []  # Define your transformations here if needed
        self._common_trsf = []  # Define your transformations here if needed

        # Class Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = list(set(self._train_targets))  # In case the class order needs to be preserved
        self._class_order = order
        logging.info(self._class_order)
        logging.info(mods)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

        self.high_snr_ref = {}
        for class_id in np.unique(self._train_targets):
            # 找出该类别在18dB SNR下的所有样本
            class_indices = np.where(self._train_targets == class_id)[0]
            max_snr = max(self._train_snr[i] for i in class_indices)
            indices = [i for i in class_indices if self._train_snr[i] == max_snr]
            ref_idx = np.random.choice(indices)
            self.high_snr_ref[class_id] = self._train_data[ref_idx]

        # 2. 创建参考信号张量（用于快速访问）
        self.ref_signals = torch.tensor(
            np.array([self.high_snr_ref[i] for i in sorted(self.high_snr_ref.keys())]),
            dtype=torch.float32
        )
        N_random_sample = 50  # 每个调制类型-SNR组合随机选取50个样本:
        # 总样本数：调制类型数*SNR数，样本中的采样数=128*50
        # 初始化随机矩阵（维度：[调制类型数*SNR数, 2, 128*50]）
        self.Random_Matrix = np.zeros([len(mods) * len(snrs), 2, 1024 * N_random_sample])
        count = 0
        # 遍历每个调制类型和SNR组合
        # mods = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        for i in range(len(mods)):
            for snr_idx in range(len(snrs)):
                # 计算当前组合在原始数据中的起始索引
                # 假设每个调制类型有len(snrs)个SNR，每个SNR有1000个样本
                # 索引起始位置，因为数据是顺序拼接的，11种调制类型，20种snr，每种都有1000个样本
                # 从每个1000种随机取出50个，放入Random_Matrix中
                # 并以[2,128*50]的形式存储下来
                start_idx = i * len(snrs) * 1000 + snr_idx * 1000
                # 从当前组合的1000个样本中随机选取N_random_sample个
                choice = np.random.choice(range(1000), size=N_random_sample, replace=False)
                # 提取随机样本并调整维度
                random_sample = X[choice + start_idx]  # 形状[50, 2, 128]
                random_sample = random_sample.swapaxes(0, 1)  # 变为[2, 50, 128]
                # 展平后两维，得到[2, 128*50]
                random_sample = np.reshape(random_sample, [2, 1024 * N_random_sample])
                # 存入随机矩阵并更新计数器
                self.Random_Matrix[count] = random_sample
                count += 1

    def _select(self, x, y, snr, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

        if isinstance(x, np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        snr_return = np.array(snr)[idxes]
        return x_return, y[idxes], snr_return

    def _select_rmm(self, x, y, snr, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        snr_return = np.array(snr)[new_idxes]

        return x[new_idxes], y[new_idxes], snr_return

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


# class DummyDataset(Dataset):
#     def __init__(self, images, labels, trsf, use_path=False):
#         assert len(images) == len(labels), "Data size error!"
#         self.images = images
#         self.labels = labels
#         self.trsf = trsf
#         self.use_path = use_path
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         if self.use_path:
#             image = self.trsf(pil_loader(self.images[idx]))
#         else:
#             image = self.trsf(Image.fromarray(self.images[idx]))
#         label = self.labels[idx]
#
#         return idx, image, label
class DummyDataset(Dataset):
    def __init__(self, images, labels, snrs, trsf=None, use_path=False):
        """
        Args:
            images (numpy.ndarray or list): 输入数据，形状为 (num_samples, height, width, channels)
            labels (numpy.ndarray or list): 标签，形状为 (num_samples,)
            trsf (callable, optional): 用于数据转换的函数，默认为 None
        """
        assert len(images) == len(labels), "Data size error!"
        assert len(snrs) == len(images), "Data size error!"
        self.images = images
        self.labels = labels
        self.snrs = snrs
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 获取样本数据
        image = self.images[idx]
        label = self.labels[idx]
        snrs = self.snrs[idx]

        # 如果提供了转换操作 (例如 ToTensor, 归一化等)
        if self.trsf:
            image = self.trsf(image)
        else:
            # 如果没有提供转换操作，则直接将数据转为 Tensor
            image = torch.tensor(image, dtype=torch.float32)

        # 返回索引、图像和标签
        return idx, image, label, snrs


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _map_new_class_index_2(y, order):
    # 构建复数标签到新索引的字典（提高查找效率）
    class_to_idx = {cls: idx for idx, cls in enumerate(order)}

    # 检查是否存在缺失标签
    missing_labels = set(y) - class_to_idx.keys()
    if missing_labels:
        raise ValueError(f"标签中存在未在order中定义的复数: {missing_labels}")

    # 通过字典直接映射（避免低效的列表.index()）
    return np.array([class_to_idx[cls] for cls in y])


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
