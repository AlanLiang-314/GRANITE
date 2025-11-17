import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2


import torchvision.transforms.functional

class FixDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        if self.transforms:
            data_item = self.transforms(data_item)
        return data_item, self.labels[idx]


class AutoencoderDataset(Dataset):
    def __init__(self, data, labels, smote_to_base, smote_to_neighbor, smote_to_alpha, origin_datalen):
        self.data = data
        self.labels = labels
        self.smote_to_base = smote_to_base
        self.smote_to_neighbor = smote_to_neighbor
        self.smote_to_alpha = smote_to_alpha
        self.origin_datalen = origin_datalen
        self.base_to_smote = self._reverse_dict(smote_to_base)
    
    @classmethod
    def _reverse_dict(self, original_dict):
        reversed_dict = defaultdict(list)
        for key, value in original_dict.items():
            reversed_dict[value].append(key)
        return dict(reversed_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleSampler(torch.utils.data.Sampler):
    def __init__(self, origin_datalen: int, generate_datalen: int,dataset_group_size: int, group_size: int, repeat_num: int, base_to_smote: dict):
        np.random.seed(1337)
        self.origin_datalen = origin_datalen
        self.generate_datalen = generate_datalen
        self.dataset_group_size = dataset_group_size
        self.group_size = group_size
        self.repeat_num = repeat_num
        indices = np.arange(self.origin_datalen)
        expanded_indices = np.tile(indices, 1)
        np.random.shuffle(expanded_indices)
        if (self.repeat_num >= self.group_size-1):
            self.repeat_num = self.group_size-1
        self.final_indices = [[idx for _ in range(self.repeat_num+1)]+[(base_to_smote[idx][i]+self.origin_datalen) for i in range(self.group_size-1 - self.repeat_num) ] for idx in expanded_indices]

        #self.final_indices = [[idx] + [idx * (self.dataset_group_size-1) + i + self.origin_datalen for i in range(self.group_size - 1)] for idx in indices]
        self.flattened_indices = [idx for batch in self.final_indices for idx in batch]
    def __iter__(self):
        return iter(self.flattened_indices)
    
    def __len__(self):
        return self.generate_datalen

class SSampler(torch.utils.data.Sampler):
    def __init__(self, final_indices):
        self.final_indices = final_indices
        self.flattened_indices = [idx for batch in self.final_indices for idx in batch]
        self.gen_len = len(self.flattened_indices)
    def __iter__(self):
        return iter(self.flattened_indices)
    def __len__(self):
        return self.gen_len


def datasets_Cifar10():
    data_mean = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
    data_std = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)

    
    transform_train = transforms.Compose(
    [transforms.ColorJitter(brightness= 0.2, contrast= 0.1, saturation=0.1, hue=0.05),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.RandomChoice([
         transforms.RandomRotation((-5,5), fill=255),
         transforms.RandomRotation((85,95), fill=255),
         transforms.RandomRotation((175,185), fill=255),
         transforms.RandomRotation((-95,-85), fill=255)
     ]),
     transforms.ToTensor(),
     transforms.Normalize(data_mean, data_std)])
    transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(data_mean, data_std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    
    def d_denorm(tensor: torch.Tensor):
        mean = torch.tensor(data_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=tensor.device).view(1, 3, 1, 1)
        
        if tensor.ndimension() == 3:  # (3, 32, 32)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return torch.clamp(tensor * std + mean, 0, 1)

    
    d_norm = transforms.Normalize(data_mean, data_std)

    return trainset, testset, d_norm, d_denorm

def datasets_Cifar100():
    data_mean = (0.5071598291397095, 0.4866936206817627, 0.44120192527770996)
    data_std = (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)
    
    transform_train = transforms.Compose(
    [transforms.ColorJitter(brightness= 0.2, contrast= 0.1, saturation=0.1, hue=0.05),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.RandomChoice([
         transforms.RandomRotation((-5,5), fill=255),
         transforms.RandomRotation((85,95), fill=255),
         transforms.RandomRotation((175,185), fill=255),
         transforms.RandomRotation((-95,-85), fill=255)
     ]),
     transforms.ToTensor(),
     transforms.Normalize(data_mean, data_std)])
    transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(data_mean, data_std)])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    
    def d_denorm(tensor: torch.Tensor):
        mean = torch.tensor(data_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=tensor.device).view(1, 3, 1, 1)
        
        if tensor.ndimension() == 3:  # (3, 32, 32)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return torch.clamp(tensor * std + mean, 0, 1)

    
    d_norm = transforms.Normalize(data_mean, data_std)

    return trainset, testset, d_norm, d_denorm

def datasets_Food101(num_classes=100):
    data_mean = (0.5071598291397095, 0.4866936206817627, 0.44120192527770996)
    data_std = (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)

    transform_train = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize(data_mean, data_std)])
    transform_test = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize(data_mean, data_std)])

    # trainset_full = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=transform_train)
    testset_full = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=transform_test)
    
    # Filter to first 100 classes
    # train_indices = [i for i, (_, label) in enumerate(trainset_full) if label < 100]
    test_indices = [i for i, (_, label) in enumerate(testset_full) if label < num_classes]
    
    # trainset = torch.utils.data.Subset(trainset_full, train_indices)
    testset = torch.utils.data.Subset(testset_full, test_indices)
    
    def d_denorm(tensor: torch.Tensor):
        mean = torch.tensor(data_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=tensor.device).view(1, 3, 1, 1)
        
        if tensor.ndimension() == 3:  # (3, 32, 32)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return torch.clamp(tensor * std + mean, 0, 1)
    
    d_norm = transforms.Normalize(data_mean, data_std)

    return None, testset, d_norm, d_denorm


import os
import pandas as pd
from PIL import Image
class TinyImageNetDataset(Dataset):
    """
    客製化的 Tiny ImageNet 資料集類別。
    
    Args:
        root_dir (str): 資料集根目錄 (包含 tiny-imagenet-200 的路徑)。
        split (str): 'train', 'val', or 'test'。
        transform (callable, optional): 應用於圖片的轉換。
    """
    def __init__(self, root_dir, split='train', transform=None, num_classes=100):
        self.root_dir = os.path.join(root_dir, 'tiny-imagenet-200/tiny-imagenet-200')
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 1. 建立類別名稱 (wnid) 到索引 (0-199) 的映射
        wnids_path = os.path.join(self.root_dir, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.wnids = [line.strip() for line in f.readlines()]
        
        self.wnids = self.wnids[:num_classes]  # 只保留前 num_classes 類別
        wnid_set = set(self.wnids)
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}

        # 2. 根據 split 載入對應的資料
        if self.split == 'train':
            train_dir = os.path.join(self.root_dir, 'train')
            for class_name in os.listdir(train_dir):
                class_dir = os.path.join(train_dir, class_name, 'images')
                if os.path.isdir(class_dir) and class_name in wnid_set:
                    class_idx = self.class_to_idx[class_name]
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
        
        elif self.split == 'val':
            val_dir = os.path.join(self.root_dir, 'val')
            val_images_dir = os.path.join(val_dir, 'images')
            annotations_path = os.path.join(val_dir, 'val_annotations.txt')
            
            # 讀取標籤檔
            val_annotations = pd.read_csv(annotations_path, sep='\t', header=None,
                                          names=['filename', 'wnid', 'x1', 'y1', 'x2', 'y2'])
            
            # 建立檔名到標籤的映射
            filename_to_label = {row['filename']: row['wnid'] for _, row in val_annotations.iterrows()}
            
            for img_name in os.listdir(val_images_dir):
                img_path = os.path.join(val_images_dir, img_name)
                class_name = filename_to_label.get(img_name)
                if class_name:
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        elif self.split == 'test':
            # 測試集沒有標籤
            test_dir = os.path.join(self.root_dir, 'test', 'images')
            for img_name in os.listdir(test_dir):
                img_path = os.path.join(test_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(-1) # 使用 -1 作為佔位符

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 確保以 RGB 格式讀取，避免灰階圖片造成維度錯誤
        image = Image.open(img_path).convert('RGB') 
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

def datasets_TinyImageNet(seed=1337, num_classes=100):
    torch.manual_seed(seed)
    data_mean = (0.4802, 0.4481, 0.3975)
    data_std = (0.2302, 0.2265, 0.2262)

    
    transform_better = v2.Compose(
        [
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=data_mean, std=data_std),
        ]
    )
    
    transform_test = v2.Compose(
    [v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),

     v2.Normalize(data_mean, data_std)])

    trainset = TinyImageNetDataset(root_dir='./data', split='train', transform=transform_better, num_classes=num_classes)
    # testset = TinyImageNetDataset(root_dir='./data', split='val', transform=transform_test)
    
    print(f"trainset len: {len(trainset)}")
    

    
    def d_denorm(tensor: torch.Tensor):
        mean = torch.tensor(data_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=tensor.device).view(1, 3, 1, 1)
        
        if tensor.ndimension() == 3:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return torch.clamp(tensor * std + mean, 0, 1)
    
    d_norm = transforms.Normalize(data_mean, data_std)

    return trainset, None, d_norm, d_denorm


def datasets_train_Cifar10(seed=1337):
    torch.manual_seed(seed)
    # data_mean = (0.5, 0.5, 0.5)
    # data_std = (0.5, 0.5, 0.5)
    data_mean = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
    data_std = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)

    
    transform_better = v2.Compose(
        [
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=data_mean, std=data_std),
        ]
    )
    
    transform_test = v2.Compose(
    [v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),

     v2.Normalize(data_mean, data_std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_better)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    print(f"trainset len: {len(trainset)}")
    
    # train_size = int(0.8 * len(trainset))
    # val_size = len(trainset) - train_size
    # trainset, valset = random_split(trainset, [train_size, val_size])

    
    def d_denorm(tensor: torch.Tensor):
        mean = torch.tensor(data_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=tensor.device).view(1, 3, 1, 1)
        
        if tensor.ndimension() == 3:  # (3, 32, 32)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return torch.clamp(tensor * std + mean, 0, 1)
    
    d_norm = transforms.Normalize(data_mean, data_std)

    return trainset, testset, d_norm, d_denorm

def datasets_train_Cifar100(seed=1337):
    torch.manual_seed(seed)
    # data_mean = (0.5, 0.5, 0.5)
    # data_std = (0.5, 0.5, 0.5)
    data_mean = (0.5071598291397095, 0.4866936206817627, 0.44120192527770996)
    data_std = (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)

    
    transform_better = v2.Compose(
        [
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=data_mean, std=data_std),
        ]
    )
    
    transform_test = v2.Compose(
    [v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),

     v2.Normalize(data_mean, data_std)])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_better)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    print(f"trainset len: {len(trainset)}")
    
    # train_size = int(0.8 * len(trainset))
    # val_size = len(trainset) - train_size
    # trainset, valset = random_split(trainset, [train_size, val_size])

    
    def d_denorm(tensor: torch.Tensor):
        mean = torch.tensor(data_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=tensor.device).view(1, 3, 1, 1)
        
        if tensor.ndimension() == 3:  # (3, 32, 32)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return torch.clamp(tensor * std + mean, 0, 1)
    
    d_norm = transforms.Normalize(data_mean, data_std)

    return trainset, testset, d_norm, d_denorm

def datasets_train_grouped_Cifar100(seed=1337, thresh="18.0", dataset="Cifar100"):
    torch.manual_seed(seed)
    # set to 0.5
    # data_mean = (0.5, 0.5, 0.5)
    # data_std = (0.5, 0.5, 0.5)
    if dataset == 'Cifar10':
        data_mean = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
        data_std = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)
    elif dataset == 'Cifar100':
        data_mean = (0.5071598291397095, 0.4866936206817627, 0.44120192527770996)
        data_std = (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)

    dataset_path = f"/trainingData/sage/alan/defence_seer/defence/datasets/{dataset}/trainset/init_2000/Norm/MSE/paper/denorm/rand_aug_1/finetune1000epoch_psnr_thresh{thresh}finetune_images_num10000.dst"
    ae_dataset = torch.load(dataset_path, weights_only=False)
    
    
    transform_better = v2.Compose(
        [
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=data_mean, std=data_std),
        ]
    )
    
    transform_test = v2.Compose(
    [v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
     v2.Normalize(data_mean, data_std)])

    if dataset == "Cifar100":
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == "Cifar10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    print(f"trainset len: {len(ae_dataset)}")
    sampler = SimpleSampler(
        ae_dataset.origin_datalen,
        len(ae_dataset),
        8,
        int(64 // 16),
        0,
        ae_dataset.base_to_smote
    )
    
    
    # train_size = int(len(sampler.final_indices) * 0.8)
    # train_indices, val_indices = sampler.final_indices[:train_size], sampler.final_indices[train_size:]
    # train_sampler = SSampler(train_indices)
    # val_sampler = SSampler(val_indices)
    train_sampler = SSampler(sampler.final_indices)
    
    ae_dataset = FixDataset(ae_dataset.data, ae_dataset.labels, transforms=transform_better)
    
    trainloader = DataLoader(
        ae_dataset,
        batch_size=64,
        sampler=train_sampler,
        pin_memory=True, num_workers=8,
    )
    
    # valloader = DataLoader(
    #     ae_dataset,
    #     batch_size=64,
    #     num_workers=4,
    #     sampler=val_sampler,
    # )
    
    testloader = DataLoader(testset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    
    def d_denorm(tensor: torch.Tensor):
        mean = torch.tensor(data_mean, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(data_std, device=tensor.device).view(1, 3, 1, 1)
        
        if tensor.ndimension() == 3:  # (3, 32, 32)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        
        return torch.clamp(tensor * std + mean, 0, 1)
    
    d_norm = transforms.Normalize(data_mean, data_std)

    
    return trainloader, testloader, d_norm, d_denorm



if __name__ == '__main__':
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    trainloader, valloader, testloader, d_norm, d_denorm = datasets_train_grouped_Cifar100()
    X, y = next(iter(trainloader))
    grid = make_grid(d_denorm(X))
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig('temp.png')
    