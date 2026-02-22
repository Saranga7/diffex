import os
import glob
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class BBC_dataset(Dataset):
    def __init__(self, root, img_size = 128, split = "train"):
    
        self.root = root
        # Store image paths
        data_path = glob.glob(os.path.join(self.root, split, "*/*"))
        # self.data = glob.glob(f"{data_path}/*")
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

        self.data = []
        for img_path in data_path:
            class_name = img_path.split(os.sep)[-2]
            self.data.append([img_path, class_name])

        # Image transformation
        self.transform = transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img_path, _class_name = self.data[index]
        _img = self.transform(Image.open(img_path))
        _label = torch.tensor(self.class_map[_class_name])
        return {"img": _img, "label": _label, "class_name": _class_name}
    
    





