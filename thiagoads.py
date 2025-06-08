import torch


def get_subset(dataset, percentage=None, num_samples=None):
    """
    Returns a subset of the dataset with the specified percentage of samples.
    """
    if percentage is None and num_samples is None:
        return dataset
    
    dataset_length = len(dataset)

    if percentage is not None:
        if not (0 < percentage <= 1):
            raise ValueError("Percentage must be between 0 and 1.")
        idxs = torch.randperm(dataset_length)[:int(dataset_length * percentage)]
    elif num_samples is not None:
        if num_samples > dataset_length:
            raise ValueError("num_samples cannot be greater than the dataset length.")
        idxs = torch.randperm(dataset_length)[:num_samples]

    return torch.utils.data.Subset(dataset, idxs)