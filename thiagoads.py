import torch
import logging

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_subset(dataset, subset_param, seed=None):
    """
    Returns a subset of the dataset based on the subset_param.
    If subset_param is None, the original dataset is returned.
    If subset_param is a float between 0 and 1, it is treated as a percentage.
    If subset_param is an integer, it is treated as the number of samples.
    The optional seed parameter is used to set the random generator seed.
    """
    if subset_param is None:
        logger.info("thiagoads: subset_param é None. Retornando o dataset original.")
        return dataset

    if seed is not None:
        logger.info(f"thiagoads: Seed {seed} foi fornecida. Definindo a semente do gerador aleatório.")
        torch.manual_seed(seed)

    dataset_length = len(dataset)

    if isinstance(subset_param, float):
        if not (0 < subset_param <= 1):
            raise ValueError("thiagoads: Percentage must be between 0 and 1.")
        logger.info(f"thiagoads: Método get_subset foi invocado para criar um subset com {subset_param * 100:.2f}% do dataset.")
        idxs = torch.randperm(dataset_length)[:int(dataset_length * subset_param)]
    elif isinstance(subset_param, int):
        if subset_param > dataset_length:
            raise ValueError("thiagoads: num_samples cannot be greater than the dataset length.")
        logger.info(f"thiagoads: Método get_subset foi invocado para criar um subset com {subset_param} exemplos.")
        idxs = torch.randperm(dataset_length)[:subset_param]
    else:
        raise TypeError("thiagoads: subset_param must be a float (percentage) or an int (number of samples).")

    logger.info(f"thiagoads: Dataset original: {dataset_length} exemplos | Subset final: {len(idxs)} exemplos.")
    return torch.utils.data.Subset(dataset, idxs)