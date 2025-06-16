import bisect
import warnings
import numpy as np, random, os
import pandas as pd
import torch
from scipy.stats import sampling
from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, TensorDataset
from collections import defaultdict


def setup_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  
    if deterministic:  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
               
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
               
                y = self.dataset.get_class(idx).item()
                if not self.dataset.classwise_indices[y]:
                    raise ValueError("No samples available for class {y}")
              
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

       
        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset[self.indices[i]][1]  
            self.classwise_indices[y.item()].append(i) 
        self.num_classes = max(self.classwise_indices.keys()) + 1
        # print("Classwise Indices:", dict(self.classwise_indices))

      
    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset[self.indices[i]][1]  # 返回目标值


class ConcatWrapper(Dataset): # TODO: Naming
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @staticmethod
    def numcls(sequence):
        s = 0
        for e in sequence:
            l = e.num_classes
            s += l
        return s

    @staticmethod
    def clsidx(sequence):
        r, s, n = defaultdict(list), 0, 0
        for e in sequence:
            l = e.classwise_indices
            for c in range(s, s + e.num_classes):
                t = np.asarray(l[c-s]) + n
                r[c] = t.tolist()
            s += e.num_classes
            n += len(e)
        return r

    def __init__(self, datasets):
        super(ConcatWrapper, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        # for d in self.datasets:
        #     assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.num_classes = self.numcls(self.datasets)
        self.classwise_indices = self.clsidx(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def get_class(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_class(sample_idx)

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

def input_tensor(data, label):
    data = np.array(data)
    data = torch.FloatTensor(data)

    label = np.array(label)
    label = label.astype(float)
    label = torch.LongTensor(label)

    return data, label

def read_excel(path):
    read_excel = pd.read_excel(path, engine='openpyxl')
    np_list = read_excel.T.to_numpy()
    data = np_list[0:-1].T
    data = np.float64(data)
    label = np_list[-1]

    data, label = input_tensor(data, label)

    return data, label

def load_dataset(root, sample='default', **kwargs):
    setup_seed(42)
    # Dataset
 
    train_val_dataset_dir = os.path.join(root, "train.xlsx")
    test_dataset_dir = os.path.join(root, "val.xlsx")

    trainset_data, trainset_targets = read_excel(train_val_dataset_dir)
    valset_data, valset_targets = read_excel(test_dataset_dir)

    trainset_dataset = TensorDataset(trainset_data, trainset_targets)
    valset_dataset = TensorDataset(valset_data, valset_targets)

    trainset = DatasetWrapper(trainset_dataset)
    valset = DatasetWrapper(valset_dataset)

    # Sampler
    if sample == 'default':
        get_train_sampler = lambda d: BatchSampler(RandomSampler(d), kwargs['batch_size'], False)
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    elif sample == 'pair':
        get_train_sampler = lambda d: PairBatchSampler(d, kwargs['batch_size'])
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    else:
        raise Exception('Unknown sampling: {}'.format(sampling))

    trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=0)
    valloader   = DataLoader(valset,   batch_sampler=get_test_sampler(valset), num_workers=0)

    return trainloader, valloader


def load_test_dataset(root, row_index=None, **kwargs):
    setup_seed(42)
    # Dataset
    test_dataset_dir = os.path.join(root, "test.xlsx")  
    testset_data, testset_targets = read_excel(test_dataset_dir)


    if row_index is not None:
        testset_data = testset_data[row_index:row_index+1]
        testset_targets = testset_targets[row_index:row_index+1]


    testset_dataset = TensorDataset(testset_data, testset_targets)
    testset = DatasetWrapper(testset_dataset)

    get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)
    testloader   = DataLoader(testset, batch_sampler=get_test_sampler(testset), num_workers=0)

    return testloader






