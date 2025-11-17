import torch

def property_scores(datapoints,prop,labels=None):
    # higher is better
    if prop == 'bright':
        scores = datapoints.mean((1,2,3))
    elif prop == 'dark':
        scores = datapoints.mean((1,2,3))
        scores = -scores
    elif prop == 'red':
        scores = datapoints.mean((2,3)) * torch.tensor([2.,-1.,-1.],device=datapoints.device).unsqueeze(0)
        scores = scores.mean((1,))
    elif prop == 'green':
        scores = datapoints.mean((2,3)) * torch.tensor([-1.,2.,-1.],device=datapoints.device).unsqueeze(0)
        scores = scores.mean((1,))
    elif prop == 'blue':
        scores = datapoints.mean((2,3)) * torch.tensor([-1.,-1.,2.],device=datapoints.device).unsqueeze(0)
        scores = scores.mean((1,))
    elif prop == 'hedge':
        gray = datapoints.mean(1)
        scores = (gray[:,1:,:] - gray[:,:-1,:]).mean((1,2))
    elif prop == 'vedge':
        gray = datapoints.mean(1)
        scores = (gray[:,:,1:] - gray[:,:,:-1]).mean((1,2))
    elif prop == 'vedge+green':
        scores_green = datapoints.mean((2,3)) * torch.tensor([-1.,2.,-1.],device=datapoints.device).unsqueeze(0)
        scores_green = scores_green.mean((1,))
        gray = datapoints.mean(1)
        scores_vedge = (gray[:,:,1:] - gray[:,:,:-1]).mean((1,2))
        scores = scores_green * 0.5 + scores_vedge * 0.5
    elif prop == 'red+car':
        scores = datapoints.mean((2,3)) * torch.tensor([2.,-1.,-1.],device=datapoints.device).unsqueeze(0)
        scores = scores.mean((1,))
        scores[labels != 1] = -1e+6
    elif prop == 'rand_conv':
        weight = torch.tensor([[[[ 0.1410,  0.1441,  0.0390],
                                 [-0.1475,  0.1789,  0.0264],
                                 [-0.1550, -0.0560,  0.1355]],

                                [[-0.1293,  0.0612,  0.0567],
                                 [ 0.1918,  0.0576,  0.1709],
                                 [-0.0039,  0.0088, -0.0689]],

                                [[-0.1303,  0.0727, -0.0907],
                                 [-0.1532, -0.0025,  0.1554],
                                 [ 0.1820,  0.0876, -0.0287]]]],device=datapoints.device)
        weight -= weight.mean()
        convolved=torch.nn.functional.conv2d(datapoints, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        scores = convolved.mean((1,2,3))
    elif prop == 'rand_conv1':
        weight = torch.tensor([[[[-0.0373, -0.0297,  0.1309],
                                 [ 0.1135, -0.0817, -0.1603],
                                 [-0.0151,  0.1849,  0.1139]],

                                [[-0.0668,  0.1902,  0.1110],
                                 [-0.1343, -0.0033, -0.0864],
                                 [-0.1639, -0.1398,  0.0666]],

                                [[ 0.0147, -0.0921,  0.1279],
                                 [-0.1277, -0.1600,  0.0268],
                                 [ 0.0624,  0.0971,  0.1694]]]],device=datapoints.device)
        weight -= weight.mean()
        convolved=torch.nn.functional.conv2d(datapoints, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        scores = convolved.mean((1,2,3))
    elif prop == 'texture_complexity':
        # 使用局部標準差來測量紋理複雜度
        # 將標準差結果通過 tanh 函數進行非線性變換
        
        # 轉換為灰度
        gray = datapoints.mean(1)
        
        # 計算局部區域的標準差（使用與平均值的差的平方）
        local_mean = torch.nn.functional.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        squared_diff = (gray - local_mean) ** 2
        local_var = torch.nn.functional.avg_pool2d(squared_diff, kernel_size=3, stride=1, padding=1)
        local_std = torch.sqrt(local_var + 1e-6)  # 加入epsilon避免數值問題
        
        # 應用非線性函數 tanh 使結果在 -1 到 1 之間
        texture_score = torch.tanh(3 * local_std).mean((1, 2))
        scores = texture_score
    elif prop == 'saturation_contrast':
        # 計算每個像素的飽和度（非線性）
        r, g, b = datapoints[:, 0, :, :], datapoints[:, 1, :, :], datapoints[:, 2, :, :]
        max_rgb = torch.max(torch.max(r, g), b)
        min_rgb = torch.min(torch.min(r, g), b)
        
        # 避免除以零
        epsilon = 1e-6
        
        # 飽和度計算 (max-min)/(max+epsilon)
        saturation = (max_rgb - min_rgb) / (max_rgb + epsilon)
        
        # 計算對比度的非線性度量
        # 使用 sigmoid 函數將值壓縮到 0-1 範圍
        mean_brightness = datapoints.mean((1, 2, 3)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = torch.sigmoid(5 * (datapoints - mean_brightness))
        contrast = contrast.mean((1, 2, 3))
        
        # 結合飽和度和對比度的非線性分數
        scores = saturation.mean((1, 2)) * contrast
    elif prop == 'color_entropy':
        # 計算色彩熵 - 測量色彩分佈的不確定性
        # 熵本身就是一個非線性測量
        
        # 簡化色彩空間 (將每個通道的值分成8個箱子)
        bins = 8
        binned = (datapoints * (bins - 1e-6)).int()
        
        # 創建色彩直方圖
        batch_size = datapoints.shape[0]
        histograms = torch.zeros((batch_size, bins**3), device=datapoints.device)
        
        # 為每個像素計算bin索引
        r, g, b = binned[:, 0], binned[:, 1], binned[:, 2]
        indices = r * bins**2 + g * bins + b
        
        # 計算每個圖像的直方圖
        for i in range(batch_size):
            histogram = torch.bincount(indices[i].flatten(), minlength=bins**3).float()
            histogram = histogram / histogram.sum()  # 標準化為概率
            
            # 計算熵 (-p*log(p))
            # 避免log(0)
            entropy = -torch.sum(histogram * torch.log2(histogram + 1e-10))
            scores[i] = entropy
    
    elif prop == 'gradient_magnitude_distribution':
        # 計算圖像梯度並使用其分佈的非線性特徵
        
        # 轉換為灰度
        gray = datapoints.mean(1)
        
        # 計算x和y方向的梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=datapoints.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=datapoints.device).float()
        
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        
        grad_x = torch.nn.functional.conv2d(gray.unsqueeze(1), sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray.unsqueeze(1), sobel_y, padding=1)
        
        # 計算梯度大小
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # 應用非線性變換：使用梯度大小的百分位數分佈
        # 90%和10%百分位數之間的比率（非線性測量）
        batch_size = gradient_magnitude.shape[0]
        scores = torch.zeros(batch_size, device=datapoints.device)
        
        for i in range(batch_size):
            flat_magnitude = gradient_magnitude[i].flatten()
            sorted_magnitude, _ = torch.sort(flat_magnitude)
            idx_90 = int(0.9 * len(sorted_magnitude))
            idx_10 = int(0.1 * len(sorted_magnitude))
            
            percentile_90 = sorted_magnitude[idx_90]
            percentile_10 = sorted_magnitude[idx_10]
            
            # 使用比率作為非線性測量
            scores[i] = torch.log(percentile_90 / (percentile_10 + 1e-6))
    return scores

def replace_orig_mod(ckpt):
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith('_orig_mod.'):
            new_key = k.replace('_orig_mod.', '')
            new_ckpt[new_key] = v
        else:
            new_ckpt[k] = v
    return new_ckpt

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

class TinyImageNetDataset(Dataset):
    """
    客製化的 Tiny ImageNet 資料集類別。
    
    Args:
        root_dir (str): 資料集根目錄 (包含 tiny-imagenet-200 的路徑)。
        split (str): 'train', 'val', or 'test'。
        transform (callable, optional): 應用於圖片的轉換。
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, 'tiny-imagenet-200/tiny-imagenet-200')
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 1. 建立類別名稱 (wnid) 到索引 (0-199) 的映射
        wnids_path = os.path.join(self.root_dir, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.wnids = [line.strip() for line in f.readlines()]
        
        self.wnids = self.wnids[:100]
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