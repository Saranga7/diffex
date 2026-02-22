import os
import logging
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pandas as pd
import glob

import torchvision.transforms.functional as Ftrans

logger = logging.getLogger(__name__)

class RandomRotate90:
    """Randomly rotate by 0°, 90°, 180°, or 270° without interpolation."""
    def __call__(self, img: Image.Image):
        k = random.choice([0, 1, 2, 3])
        if k == 0:
            return img
        ops = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}
        return img.transpose(ops[k])


class SmalaDataset(Dataset):
    def __init__(self, path, image_size, transform = None, exts=(".tif", ".tiff", ".TIF", ".TIFF"), exclude_list: str = None):
        """
        root_dir: folder with images
        exclude_list: optional path to a text file with one bad filepath per line (full paths or relative to root_dir)
        """
        self.root_dir = Path(path)
        # gather files first
        files = sorted([p for p in self.root_dir.rglob("*") if p.suffix in exts])

        if transform is None:
            transform = transforms.Compose(
            [   
                RandomRotate90(),
                transforms.CenterCrop(size = (512, 512)),
                transforms.Resize(size = (image_size, image_size)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomVerticalFlip(p = 0.5),
                # transforms.RandomApply([transforms.ColorJitter(brightness = 0.1, contrast = 0.3, saturation = 0.3, hue = 0.3)], p = 0.4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            ]
        )
            
        self.transform = transform

        # build a set of excluded paths (normalized strings)
        exclude_set = set()
        if exclude_list:
            exclude_path = Path(exclude_list)
            if exclude_path.exists():
                with open(exclude_path, "r") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        # normalize: try absolute, then relative
                        p_abs = str(Path(s))
                        p_rel = str(self.root_dir / s) if not Path(s).is_absolute() else p_abs
                        exclude_set.add(p_abs)
                        exclude_set.add(p_rel)
            else:
                logger.warning(f"exclude_list file not found: {exclude_list} -- continuing without excludes")

        # filter files
        self.files = [p for p in files if str(p) not in exclude_set]
        if exclude_set:
            logger.info(f"Excluded {len(files) - len(self.files)} files from dataset using {exclude_list}")

        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p)
        if self.transform:
            img = self.transform(img)

        return {"img": img, "index": idx}


class BBCDataset(Dataset):
    def __init__(self, path,
                img_size = 128, 
                split = None,
                as_tensor: bool = True,
                do_augment: bool = True,
                do_normalize: bool = True):
    
        self.path = path
        # Store image paths
        if split:
            data_path = glob.glob(os.path.join(self.path, split, "*/*"))
        else:
            data_path = glob.glob(os.path.join(self.path, "*/*/*"))

        self.class_map = {}
        self.class_distribution = {}

        for img_path in data_path:
            class_name = img_path.split(os.sep)[-2]
            if class_name not in self.class_distribution:
                self.class_distribution[class_name] = 1
            else:
                self.class_distribution[class_name] +=1

        print("Class distribution: ", self.class_distribution)
        for index, entity in enumerate(self.class_distribution):
            self.class_map[entity] = index
        
        # self.data = [path for path in data_path]
        self.data = []
        for img_path in data_path:
            class_name = img_path.split(os.sep)[-2]
            self.data.append([img_path, class_name])
   
        # Image transformation
        transform = [
           transforms.Resize((img_size, img_size)),
        ]

        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                )
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        assert idx < len(self.data)
        img_path, _class_name = self.data[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.class_map[_class_name])

        return {'img': img, 'index': idx, 'label': label}
    

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg'],
        do_augment: bool = True,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=False,
        has_subdir: bool = True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        # relative paths (make it shorter, saves memory and faster to sort)
        if has_subdir:
            self.paths = [
                p.relative_to(folder) for ext in exts
                for p in Path(f'{folder}').glob(f'**/*.{ext}')
            ]
        else:
            self.paths = [
                p.relative_to(folder) for ext in exts
                for p in Path(f'{folder}').glob(f'*.{ext}')
            ]
        if sort_names:
            self.paths = sorted(self.paths)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]
    


if __name__ == "__main__":
    # test the dataset
    data = SmalaDataset(path = '/projects/smala3/Saranga/preprocessed_data/subneg_reacquired', image_size = 256)
    print(f"dataset length: {len(data)}")
    for i in range(10):
        item = data[i]
        print(f"item {i} img shape: {item['img'].shape}")

    
    loader = DataLoader(data, batch_size = 8, shuffle = True, num_workers = 4)
    for batch in loader:
        print(f"batch img shape: {batch['img'].shape}")
        break