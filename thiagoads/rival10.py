import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import os
import glob
import json
import random
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
import pickle
from binascii import a2b_base64
from tqdm import tqdm
import requests  # Adicionado para realizar o download
import zipfile

# UPDATE _DATA_ROOT to '{path to dir where rival10.zip is unzipped}/RIVAL10/'
# _DATA_ROOT = None
# _LABEL_MAPPINGS = './datasets/label_mappings.json'
# _WNID_TO_CLASS = './datasets/wnid_to_class.json'

_ALL_ATTRS = ['long-snout', 'wings', 'wheels', 'text', 'horns', 'floppy-ears',
              'ears', 'colored-eyes', 'tail', 'mane', 'beak', 'hairy', 
              'metallic', 'rectangular', 'wet', 'long', 'tall', 'patterned']

def attr_to_idx(attr):
    return _ALL_ATTRS.index(attr)

def idx_to_attr(idx):
    return _ALL_ATTRS[idx]

def resize(img): 
    return np.array(Image.fromarray(np.uint8(img)).resize((224,224))) / 255

def to_3d(img):
    return np.stack([img, img, img], axis=-1)

def save_uri_as_img(uri, fpath='tmp.png'):
    ''' saves raw mask and returns it as an image'''
    binary_data = a2b_base64(uri)
    with open(fpath, 'wb') as f:
        f.write(binary_data)
    img = mpimg.imread(fpath)
    img = resize(img)
    # binarize mask
    img = np.sum(img, axis=-1)
    img[img != 0] = 1
    img = to_3d(img)
    return img


"""
Original Dataset() class for the RIVAL-10 dataset
Refer to https://github.com/mmoayeri/RIVAL10/blob/gh-pages/datasets/local_rival10.py 
"""
class LocalRIVAL10(Dataset):
    def __init__(self, root, train=True, masks_dict=True, include_aug=False, apply_transform=True):
        '''
        Set masks_dict to be true to include tensor of attribute segmentations when retrieving items.
        Set apply_transform to False to return raw PIL images instead of transformed images.
        '''
        self.train = train
        self.apply_transform = apply_transform  # Added apply_transform flag
        self.data_root = os.path.join(root, 'train' if self.train else 'test') + '/'  # Adicionado '/' ao final
        self.masks_dict = masks_dict

        self.instance_types = ['ordinary']
        # NOTE: 
        if include_aug:
            self.instance_types += ['superimposed', 'removed']
        
        self.instances = self.collect_instances()
        self.resize = transforms.Resize((224,224))

        label_mappings_path = os.path.join(root, 'meta', 'label_mappings.json')
        wnid_to_class_path = os.path.join(root, 'meta', 'wnid_to_class.json')

        with open(label_mappings_path, 'r') as f:
            self.label_mappings = json.load(f)
        with open(wnid_to_class_path, 'r') as f:
            self.wnid_to_class = json.load(f)

    def get_rival10_og_class(self, img_url):
        wnid = img_url.split('/')[-1].split('_')[0]
        inet_class_name = self.wnid_to_class[wnid]
        classname, class_label = self.label_mappings[inet_class_name]
        return classname, class_label

    def collect_instances(self):
        self.instances_by_type = dict()
        self.all_instances = []
        for subdir in self.instance_types:
            instances = []
            dir_path = self.data_root + subdir
            for f in tqdm(glob.glob(dir_path+'/*')):
                if '.JPEG' in f and 'merged_mask' not in f:
                    img_url = f
                    label_path = f[:-5] + '_attr_labels.npy'
                    merged_mask_path = f[:-5] + '_merged_mask.JPEG'
                    mask_dict_path = f[:-5] + '_attr_dict.pkl'
                    instances.append((img_url, label_path, merged_mask_path, mask_dict_path))
            self.instances_by_type[subdir] = instances.copy()
            self.all_instances.extend(self.instances_by_type[subdir])

    def __len__(self):
        return len(self.all_instances)

    def transform(self, imgs):
        transformed_imgs = []
        i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.8,1.0),ratio=(0.75,1.25))
        coin_flip = (random.random() < 0.5)
        for ind, img in enumerate(imgs):
            if self.train:
                img = TF.crop(img, i, j, h, w)

                if coin_flip:
                    img = TF.hflip(img)

            img = TF.to_tensor(self.resize(img))
            
            if img.shape[0] == 1:
                img = torch.cat([img, img, img], axis=0)
            
            transformed_imgs.append(img)

        return transformed_imgs

    def merge_all_masks(self, mask_dict):
        merged_mask = np.zeros((224,224,3))
        for attr in mask_dict:
            if attr == 'entire-object':
                continue
            mask_uri = mask_dict[attr]
            mask = save_uri_as_img(mask_uri)
            merged_mask = mask if merged_mask is None else mask + merged_mask
        merged_mask[merged_mask > 0] = 1
        return merged_mask

    def __getitem__(self, i):
        '''
        Returns dict with following keys:
            img
            attr_labels: binary vec with 1 for present attrs
            changed_attr_labels: binary vec with 1 for attrs that were removed or pasted (not natural)
            merged_mask: binary mask with 1 for any attribute region
            attr_masks: tensor w/ mask per attribute. Masks are empty for non present attrs
        '''
        img_url, label_path,  merged_mask_path, mask_dict_path = self.all_instances[i]

        # get rival10 info for original image (label may not hold for attr-augmented images)
        class_name, class_label = self.get_rival10_og_class(img_url)

        # load img
        img = Image.open(img_url)
        if img.mode == 'L':
            img = img.convert("RGB")

        # load labels
        labels = np.load(label_path)
        attr_labels = torch.Tensor(labels[0]).long()
        changed_attrs = torch.Tensor(labels[1]).long() # attrs that were added or removed

        merged_mask_img = Image.open(merged_mask_path)
        imgs = [img, merged_mask_img]
        if self.masks_dict:
            try:
                with open(mask_dict_path, 'rb') as fp:
                    mask_dict = pickle.load(fp)
            except:
                mask_dict = dict()
            for attr in mask_dict:
                mask_uri = mask_dict[attr]
                mask = save_uri_as_img(mask_uri)
                imgs.append(Image.fromarray(np.uint8(255*mask)))

        if self.apply_transform:  # Apply transformation if flag is True
            transformed_imgs = self.transform(imgs)
            img = transformed_imgs.pop(0)
            merged_mask = transformed_imgs.pop(0)
        else:
            merged_mask = merged_mask_img  # Keep as PIL image

        out = dict({'img':img, 
                    'attr_labels': attr_labels, 
                    'changed_attrs': changed_attrs,
                    'merged_mask' :merged_mask,
                    'og_class_name': class_name,
                    'og_class_label': class_label})
        if self.masks_dict:
            attr_masks = [torch.zeros(img.shape) for i in range(len(_ALL_ATTRS)+1)]
            if self.apply_transform:
                for i, attr in enumerate(mask_dict):
                    ind = -1 if attr == 'entire-object' else attr_to_idx(attr)
                    attr_masks[ind] = transformed_imgs[i]
            out['attr_masks'] = torch.stack(attr_masks)
        
        return out
    

class RIVAL10(Dataset):
    def __init__(self, root = "clean_data/rival10", train = True, transform = None, download = True):
        # Cria a pasta root e subpastas caso não existam
        os.makedirs(root, exist_ok=True)
        dataset_zip_path = os.path.join(root, "rival10.zip")
        dataset_extract_path = os.path.join(root, "RIVAL10")

        # Realiza o download do arquivo zip do dataset, se necessário
        if download and not os.path.exists(dataset_zip_path):
            print("Downloading dataset...")
            url = "https://app.box.com/index.php?rm=box_download_shared_file&shared_name=iflviwl5rbdgtur1rru3t8f7v2vp0gww&file_id=f_944375052992"
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(dataset_zip_path, "wb") as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print("Download complete.")

        # Extrai o arquivo zip, se necessário
        if not os.path.exists(dataset_extract_path):
            print("Extracting dataset...")
            with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
                for file in tqdm(zip_ref.infolist(), desc="Extracting", unit="file"):
                    zip_ref.extract(file, root)
            print("Extraction complete.")

        self.dataset = LocalRIVAL10(root=dataset_extract_path, train=train, masks_dict=False, include_aug=False, apply_transform=False)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img = data['img']
        if self.transform:
            img = self.transform(img)
        label = data['og_class_label']
        return img, label

def get_rival10_mean_and_std(image_size=32):
    mean_std_values = {
        224: {
            "mean": (0.4810, 0.4733, 0.4248),
            "std": (0.2569, 0.2518, 0.2725)
        },
        32: {
            "mean": (0.4811, 0.4733, 0.4249),
            "std": (0.2343, 0.2293, 0.2523)
        }
    }

    if image_size not in mean_std_values:
        raise ValueError(f"Mean and std values for image_size={image_size} are not registered.")

    return mean_std_values[image_size]["mean"], mean_std_values[image_size]["std"]

# if __name__ == "__main__":

    # print(get_rival10_mean_and_std())
    # print(get_rival10_mean_and_std(image_size=32))
    # print(get_rival10_mean_and_std(image_size=224))

    # from torch.utils.data import DataLoader
    # from torchvision.transforms import ToTensor, Compose, Resize

    # transform = transforms.Compose([
    #     transforms.Resize(32), # retirar em caso de querer calcular pro dataset original
    #     transforms.ToTensor()
    # ])

    # # Dataset RIVAL10
    # dataset = RIVAL10(root="clean_data/rival10", train=True, transform=transform, download=False)
    # loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # # Inicializa listas para acumular valores
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # channel_sum = torch.zeros(3).to(device)
    # channel_squared_sum = torch.zeros(3).to(device)
    # total_pixels = 0

    # for images, _ in loader:
    #     images = images.to(device)
    #     batch_samples = images.size(0)
    #     total_pixels += batch_samples * images.size(2) * images.size(3)
        
    #     # Soma acumulada dos valores dos canais
    #     channel_sum += images.sum(dim=(0, 2, 3))
    #     channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))

    # # Calcula média e desvio padrão
    # mean = channel_sum / total_pixels
    # std = torch.sqrt(channel_squared_sum / total_pixels - mean ** 2)

    # print(f"Mean: {mean}")
    # print(f"Std: {std}")

    # Resize(224), ToTensor(), apply_transform = false
    # Mean: tensor([0.4810, 0.4733, 0.4248], device='cuda:0')
    # Std: tensor([0.2569, 0.2518, 0.2725], device='cuda:0')

    # Resize(32), ToTensor(), apply_transform = false
    # Mean: tensor([0.4811, 0.4733, 0.4249], device='cuda:0')
    # Std: tensor([0.2343, 0.2293, 0.2523], device='cuda:0')