import torch
import logging

from torchvision.models import mobilenet_v3_small, shufflenet_v2_x0_5

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
        logger.info("subset_param é None. Retornando o dataset original.")
        return dataset

    if seed is not None:
        logger.info(f"Seed {seed} foi fornecida. Definindo a semente do gerador aleatório.")
        torch.manual_seed(seed)

    dataset_length = len(dataset)

    if isinstance(subset_param, float):
        if not (0 < subset_param <= 1):
            raise ValueError("thiagoads: Percentage must be between 0 and 1.")
        logger.info(f"Método get_subset foi invocado para criar um subset com {subset_param * 100:.2f}% do dataset.")
        idxs = torch.randperm(dataset_length)[:int(dataset_length * subset_param)]
    elif isinstance(subset_param, int):
        if subset_param > dataset_length:
            raise ValueError("num_samples cannot be greater than the dataset length.")
        logger.info(f"Método get_subset foi invocado para criar um subset com {subset_param} exemplos.")
        idxs = torch.randperm(dataset_length)[:subset_param]
    else:
        raise TypeError("thiagoads: subset_param must be a float (percentage) or an int (number of samples).")

    logger.info(f"Dataset original: {dataset_length} exemplos | Subset final: {len(idxs)} exemplos.")
    return torch.utils.data.Subset(dataset, idxs)


def get_mobilenet_v3_small_model(num_classes, channels, droprate=0.005):
    """
    Creates a MobileNetV3 Small model with the specified number of classes and input channels.

    Args:
        num_classes (int): The number of output classes for the model.
        channels (int): The number of input channels. If set to 1, the model will be adjusted for grayscale input.
        droprate (float, optional): Dropout rate for the model. Defaults to 0.005.

    Returns:
        torch.nn.Module: A MobileNetV3 Small model configured with the specified parameters.
    """
    logger.info(f"Criando MobileNetV3 Small model com {num_classes} classes e {channels} input channels.")
    mobilenet_v3_small_model = mobilenet_v3_small(num_classes=num_classes, width_mult=1.0)
    if channels == 1:  # Ajustar para grayscale
        mobilenet_v3_small_model.features[0][0] = torch.nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
    return mobilenet_v3_small_model

def get_shufflenet_v2_x0_5_model(num_classes, channels, droprate=0.005):
    """
    Creates a ShuffleNet model with the specified number of classes and input channels.

    Args:
        num_classes (int): The number of output classes for the model.
        channels (int): The number of input channels. If set to 1, the model will be adjusted for grayscale input.
        droprate (float, optional): Dropout rate for the model. Defaults to 0.005.

    Returns:
        torch.nn.Module: A ShuffleNet model configured with the specified parameters.
    """
    logger.info(f"Criando ShuffleNet model com {num_classes} classes e {channels} input channels.")
    from torchvision.models import shufflenet_v2_x0_5
    shufflenet_model = shufflenet_v2_x0_5(num_classes=num_classes)
    if channels == 1:  # Ajustar para grayscale
        shufflenet_model.conv1[0] = torch.nn.Conv2d(channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
    return shufflenet_model