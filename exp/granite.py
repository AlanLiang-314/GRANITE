import torch
from models.model import *
import random
import torch
from torch.utils.data import DataLoader
from utils.simple_breach import run_metrics
from utils.utils import property_scores, replace_orig_mod
from dataloader.dataloader import *
from tqdm import tqdm
from models.resnet_init import ResNet18
import os
import argparse
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class GraniteConfig:
    dataset: str = 'Cifar100'  # 'Cifar10' or 'Cifar100'
    loader_type: str = 'testloader'  # 'testloader' or 'randomloader'
    batch_size: int = 64
    seen_examples: int = 20
    seed: int = 42
    par_sel_seed: int = 98
    par_sel_num: int = 8400
    par_sel_frac: int = 0.001
    input_shape: tuple = (3, 32, 32)
    num_classes: int = 10
    device: str = 'cuda:0'

def cv(a):
    return np.std(a) / np.mean(a) if np.mean(a) != 0 else np.nan

class RandomImageDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, input_shape, num_classes, d_norm, seed):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.d_norm = d_norm
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        random_image = torch.randn(*self.input_shape)
        normalized_image = random_image
        # normalized_image = self.d_norm(random_image)
        random_label = torch.randint(0, self.num_classes, (1,)).item()
        return normalized_image, random_label


def load_malicious(attack_prop='bright', config: GraniteConfig = GraniteConfig(), seed=42, batch_size=64, par_sel_num=8400, par_sel_frac=0.001, par_sel_seed=98, weight_path=None):
    if weight_path is None:
        checkpoint_path = f"seer_weights/{config.dataset}/B{config.batch_size}C1{attack_prop}{config.dataset}Epoch1000.params"
    else:
        checkpoint_path = weight_path
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    public_model = ResNet18(seed=seed, num_classes=config.num_classes)
    par_sel=ParamSelector(public_model, par_sel_num, par_sel_frac, sparse_grad=False, seed=par_sel_seed)
    grad_ex=GradientExtractor(public_model, par_sel)
    input_shape = config.input_shape
    reconstructor=torch.nn.Linear(par_sel.num_par, input_shape[0]*input_shape[1]*input_shape[2], bias=True)
    public_model.load_state_dict(checkpoint['public_model_state_dict'])
    reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
    
    return public_model, grad_ex, reconstructor

def load_clean(weight_path = None, init_type='default', config: GraniteConfig = GraniteConfig()):
    clean_model = ResNet18(seed=config.seed, num_classes=config.num_classes, init_type=init_type)
    clean_par_sel = ParamSelector(clean_model, config.par_sel_num, config.par_sel_frac, sparse_grad=False, seed=config.par_sel_seed)
    clean_grad_ex = GradientExtractor(clean_model, clean_par_sel)
    
    if weight_path is None:
        return clean_model, clean_grad_ex
    
    clean_ckpt = torch.load(weight_path, map_location=torch.device('cpu'))
    clean_ckpt = replace_orig_mod(clean_ckpt)
    clean_model.load_state_dict(clean_ckpt)

    return clean_model, clean_grad_ex


class GRANITE():
    def __init__(self, public_model, public_grad_ex, reconstructor=None, config: GraniteConfig = GraniteConfig()):
        self.pub_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.config: GraniteConfig = config
        self.public_model = public_model
        self.public_grad_ex = public_grad_ex
        self.reconstructor = reconstructor
        
        if config.dataset == 'Cifar10':
            _, testset, d_norm, self.d_denorm = datasets_Cifar10()
        elif config.dataset == 'Cifar100':
            _, testset, d_norm, self.d_denorm = datasets_Cifar100()
        elif config.dataset == 'TinyImageNet':
            testset, _, d_norm, self.d_denorm = datasets_TinyImageNet(num_classes=config.num_classes)
        elif config.dataset == 'Food101':
            _, testset, d_norm, self.d_denorm = datasets_Food101(num_classes=config.num_classes)

        g = torch.Generator()
        g.manual_seed(config.seed)
        if config.loader_type == 'testloader':
            sampler = torch.utils.data.RandomSampler(testset, replacement=True, num_samples=int(1e9), generator=g)
            self.dataloader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=4, sampler=sampler, pin_memory=True)
        elif config.loader_type == 'randomloader':
            random_dataset = RandomImageDataset(num_samples=int(1e7), input_shape=self.config.input_shape, num_classes=self.config.num_classes, d_norm=d_norm, seed=config.seed)
            self.dataloader = DataLoader(random_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        self.public_model, self.public_grad_ex = self.public_model.to(config.device), self.public_grad_ex.to(config.device)
        
        if self.reconstructor is not None:
            self.reconstructor = self.reconstructor.to(config.device)

    def get_bdW(self, public_model, pub_loss_fn, grad_ex, X, y):
        batch_public_loss=pub_loss_fn(public_model(X), y)
        
        batch_public_gradients = torch.autograd.grad(
            outputs=batch_public_loss,
            inputs=public_model.parameters(),
            grad_outputs=torch.eye(self.config.batch_size, device=self.config.device),
            is_grads_batched=True,
            allow_unused=False
        )
        
        with torch.inference_mode():
            batch_dWflat=torch.cat([l.flatten(start_dim=1) for l in batch_public_gradients],dim=1)
            bdW = grad_ex.par_sel(batch_dWflat, single_grad=False, flat_cat=True)
            
        return batch_dWflat, bdW

    def get_norm_results(self, attack_prop='bright', enable_tqdm=False):
        norms_log = []
        psnrs = []
        self.image_outs = []
        for i, (X, y) in tqdm(enumerate(self.dataloader), total=self.config.seen_examples, desc='Evaluating grad norms', disable=not enable_tqdm):
            X, y = X.to(self.config.device), y.to(self.config.device)
            if X.shape != (self.config.batch_size, *self.config.input_shape):
                continue
            
            scores_batch = property_scores(X, attack_prop)
            order = torch.argsort(scores_batch, descending=True)
            X, y, scores_batch = X[order], y[order], scores_batch[order]
            
            batch_dWflat, bdW = self.get_bdW(self.public_model, self.pub_loss_fn, self.public_grad_ex, X, y) 
            with torch.no_grad():
                bdW = torch.sum(bdW, dim=0)
                batch_dWflat_norm = torch.linalg.vector_norm(batch_dWflat, dim=1)
                
                if (self.reconstructor is not None) and (self.config.loader_type != 'randomloader'):
                    reconstructor_o = self.reconstructor(bdW).reshape(*self.config.input_shape)
                    reconstructor_o = self.d_denorm(reconstructor_o).detach().cpu()
                    metrics = run_metrics(reconstructor_o.unsqueeze(0), self.d_denorm(X))
                    # origin = self.d_denorm(X[metrics['selector']]).squeeze().detach().cpu()
                    psnrs.append(metrics['max_psnr'])
                    self.image_outs.append(reconstructor_o)
                    # image_outs.append(origin)
                
            if i >= self.config.seen_examples:
                break
            
            norms_log.append(batch_dWflat_norm.detach().cpu().numpy()) # (batch_size,)
            
        norm_cvs = list(map(cv, norms_log))
        self.norm_cvs = norm_cvs
        
        norm_cvs_mean = np.mean(norm_cvs)
        norm_cvs_std = np.std(norm_cvs)
        
        return norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs
    


if __name__ == '__main__':
    config = GraniteConfig(
        dataset='Cifar100',
        loader_type='randomloader',
        batch_size=64,
        seen_examples=20,
        seed=42,
        par_sel_seed=98,
        par_sel_num=8400,
        par_sel_frac=0.001,
        num_classes=100,
        input_shape=(3, 32, 32),
        device='cuda:1'
    )
    
    model, grad_ex = load_clean(config=config)
    granite = GRANITE(model, grad_ex, config=config)
    ref_norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop='green', enable_tqdm=True)
    print(f'Reference Model - Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    
    model, grad_ex, reconstructor = load_malicious(attack_prop='green', config=config)
    granite_mal = GRANITE(model, grad_ex, reconstructor=reconstructor, config=config)
    mal_norm_cvs, mal_norm_cvs_mean, mal_norm_cvs_std, mal_psnrs = granite_mal.get_norm_results(attack_prop='green', enable_tqdm=True)
    print(f'Malicious Model - Norm CVs Mean: {mal_norm_cvs_mean}, Std: {mal_norm_cvs_std}')
    
    
    # # TinyImageNet helper returns (trainset, None, d_norm, d_denorm); use trainset here
    # # trainset, _, d_norm, d_denorm = datasets_TinyImageNet(num_classes=config.num_classes)
    # _, trainset, d_norm, d_denorm = datasets_Cifar10()
    # sampler = torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=int(1e9))
    # dataloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=False, num_workers=4, sampler=sampler)
    # loader_iter = iter(dataloader)
    
    # results = {}
    # public_model, grad_ex = load_clean(config=config)
    # public_model.to(config.device)
    # optimizer = torch.optim.SGD(public_model.parameters(), lr=0.01, momentum=0.9)
    
    # k = 150
    # loss_fn = torch.nn.CrossEntropyLoss()
    # for step in range(k):
    #     X, y = next(loader_iter)
    #     X, y = X.to(config.device), y.to(config.device)
        
    #     optimizer.zero_grad()
    #     loss = loss_fn(public_model(X), y)
    #     loss.backward()
    #     optimizer.step()
        
    # torch.save(public_model.state_dict(), f'weights/Cifar10/reference_model_class{config.num_classes}.params')
    
    
    # config = GraniteConfig(
    #     dataset='Cifar100',
    #     loader_type='testloader',
    #     batch_size=64,
    #     seen_examples=300, #300
    #     seed=42,
    #     par_sel_seed=98,
    #     par_sel_num=8400,
    #     par_sel_frac=0.001,
    #     num_classes=100,
    #     input_shape=(3, 32, 32),
    #     device='cuda:0'
    # )


    # model, grad = load_clean(config=config)
    # reconstructor = torch.nn.Linear(grad.par_sel.num_par, 3 * 32 * 32, bias=True)
    # granite = GRANITE(model, grad, reconstructor, config=config)
    # ref_norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop='green', enable_tqdm=True)
    # print(f'Reference Model - Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    # import pickle
    # with open('results/experiment_11_results_extend.pkl', 'wb') as f:
    #     pickle.dump({
    #         'ref_norm_cvs': ref_norm_cvs,
    #         'norm_cvs_mean': norm_cvs_mean,
    #         'norm_cvs_std': norm_cvs_std,
    #         'psnrs': psnrs
    #     }, f)
    
