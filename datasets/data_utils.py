import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler

from .DistributedProxySampler import DistributedProxySampler


def split_cissl_data(
    data,
    targets,
    max_labeled_per_class,
    max_unlabeled_per_class,
    lb_imb_ratio,
    ulb_imb_ratio,
    imb_type,
    num_classes,
    include_lb_to_ulb=True,
    seed=0,
):
    """
    data & target is splitted into labeled and unlabeld data.
    (for class-imbalance setting) FIXME: add description

    Args
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
        seed: Get deterministic results of labeled and unlabeld data
    """
    state = np.random.get_state()
    np.random.seed(seed)

    data, targets = np.array(data), np.array(targets)

    lb_data, lb_targets, lb_idx, lb_class_num_list = sample_imb_data(
        data, targets, max_labeled_per_class, lb_imb_ratio, imb_type, num_classes
    )
    rst_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data

    ulb_data, ulb_targets, ulb_idx, ulb_class_num_list = sample_imb_data(
        data[rst_idx], targets[rst_idx], max_unlabeled_per_class, ulb_imb_ratio, imb_type, num_classes
    )
    ulb_idx = rst_idx[ulb_idx]  # correct the ulb_idx

    all_idx = np.concatenate([lb_idx, ulb_idx])
    assert np.unique(all_idx).shape == all_idx.shape  # check no duplicate value

    # show the number of data in each class
    print(f"#labeled  : {lb_class_num_list.sum()}, {lb_class_num_list}")
    print(f"#unlabeled: {ulb_class_num_list.sum()}, {ulb_class_num_list}")

    np.random.set_state(state)

    if include_lb_to_ulb:
        ulb_data = np.concatenate((lb_data, ulb_data), axis=0)
        ulb_targets = np.concatenate((lb_targets, ulb_targets), axis=0)

    return lb_data, lb_targets, ulb_data, ulb_targets


def sample_imb_data(data, targets, max_labels_per_class, imb_ratio, imb_type, num_classes, seed=None):
    """
    data & target is splitted into labeled and unlabeld data.
    (for class-imbalance setting) FIXME: add description

    Args
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
        seed: Get deterministic results of labeled and unlabeld data
    """
    state = None
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    data, targets = np.array(data), np.array(targets)
    class_num_list = gen_imb_list(max_labels_per_class, imb_ratio, imb_type, num_classes)
    _data, _targets, _idx = sample_data(data, targets, class_num_list)

    if state is not None:
        np.random.set_state(state)

    return _data, _targets, _idx, class_num_list


def sample_data(data, target, class_num_list):
    """
    FIXME: add description
    samples for labeled data
    (sampling with imbalanced ratio over classes)
    """
    lb_data, lbs, lb_idx = [], [], []
    for c in range(len(class_num_list)):
        idx = np.where(target == c)[0]
        assert len(idx) >= class_num_list[c], "insufficient unlabeled data to select"
        idx = np.random.choice(idx, class_num_list[c], False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)


def gen_imb_list(max_labels_per_class, imb_ratio, imb_type, num_classes):
    # FIXME: add description
    imb_ratio_lt1 = imb_ratio < 1
    if imb_ratio_lt1:
        imb_ratio = 1 / imb_ratio

    if imb_type == "long":
        mu = np.power(1 / imb_ratio, 1 / (num_classes - 1))
        class_num_list = []
        for i in range(num_classes):
            if i == (num_classes - 1):
                class_num_list.append(int(max_labels_per_class / imb_ratio))
            else:
                class_num_list.append(int(max_labels_per_class * np.power(mu, i)))

    if imb_type == "step":
        class_num_list = []
        for i in range(num_classes):
            if i < int(num_classes / 2):
                class_num_list.append(int(max_labels_per_class))
            else:
                class_num_list.append(int(max_labels_per_class / imb_ratio))

    if imb_ratio_lt1:
        class_num_list.reverse()

    class_num_list = np.array(class_num_list).astype(int)

    return class_num_list


def get_sampler_by_name(name):
    """
    get sampler in torch.utils.data.sampler by name
    """
    sampler_name_list = sorted(
        name for name in torch.utils.data.sampler.__dict__ if not name.startswith("_") and callable(sampler.__dict__[name])
    )
    try:
        if name == "DistributedSampler":
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)

    except Exception as e:
        print(repr(e))
        print("[!] select sampler in:\t", sampler_name_list)


def get_data_loader(
    dset,
    batch_size=None,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    data_sampler=None,
    replacement=True,
    num_epochs=None,
    num_iters=None,
    generator=None,
    drop_last=True,
    distributed=False,
):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.

    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs
        num_iters: total batch -> num_iters
    """

    assert batch_size is not None

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)

        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
        else:
            num_replicas = 1

        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset) * num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters * num_replicas
        else:
            num_samples = len(dset)

        if data_sampler.__name__ == "RandomSampler":
            data_sampler = data_sampler(dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")

        if distributed:
            """
            Different with DistributedSampler,
            the DistribuedProxySampler does not shuffle the data (just wrapper for dist).
            """
            data_sampler = DistributedProxySampler(data_sampler)

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot
