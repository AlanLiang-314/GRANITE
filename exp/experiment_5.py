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



def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = False
set_all_seeds(42) 

def cv(a):
    return np.std(a) / np.mean(a) if np.mean(a) != 0 else np.nan

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = 'Cifar100'
attack_prop = 'green'
loader_type = 'randomloader'  # 'testloader' or 'randomloader'
task = 'malicious'  # 'malicious' or 'clean'
seen_examples = 50

seed = 1337
par_sel_seed = 98
par_sel_num = 8400
par_sel_frac = 0.001
batch_size = 64

if dataset == 'Cifar10':
    input_shape = (3, 32, 32)
    num_classes = 10
elif dataset == 'Cifar100':
    input_shape = (3, 32, 32)
    num_classes = 100


def load_malicious(dataset, attack_prop, batch_size=64, num_classes=10, input_shape=(3,32,32)):
    checkpoint_path = f"seer_weights/{dataset}/B{batch_size}C1{attack_prop}{dataset}Epoch1000.params"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    public_model = ResNet18(seed=seed, num_classes=num_classes).to(device)
    par_sel=ParamSelector(public_model, par_sel_num, par_sel_frac, sparse_grad=False, seed=par_sel_seed).to(device)
    grad_ex=GradientExtractor(public_model, par_sel).to(device)
    reconstructor=torch.nn.Linear(par_sel.num_par, input_shape[0]*input_shape[1]*input_shape[2], bias=True).to(device)
    public_model.load_state_dict(checkpoint['public_model_state_dict'])
    reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
    
    return public_model, grad_ex, reconstructor

def load_clean(weight_path = None, init_type='default', num_classes=10, seed=1337):
    clean_model = ResNet18(seed=seed, num_classes=num_classes, init_type=init_type).to(device)
    clean_par_sel = ParamSelector(clean_model, par_sel_num, par_sel_frac, sparse_grad=False, seed=par_sel_seed).to(device)
    clean_grad_ex = GradientExtractor(clean_model, clean_par_sel).to(device)
    
    if weight_path is None:
        return clean_model, clean_grad_ex
    
    clean_ckpt = torch.load(weight_path, map_location=torch.device('cpu'))
    clean_ckpt = replace_orig_mod(clean_ckpt)
    clean_model.load_state_dict(clean_ckpt)

    return clean_model, clean_grad_ex


class RandomImageDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, input_shape, num_classes, d_norm):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.d_norm = d_norm

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        random_image = torch.randn(*self.input_shape)
        normalized_image = self.d_norm(random_image)
        random_label = torch.randint(0, self.num_classes, (1,)).item()
        return normalized_image, random_label


class GRANITE():
    def __init__(self, dataset, public_model, public_grad_ex, reconstructor = None, loader_type='testloader', seen_examples=20, n_sigma=3):
        self.pub_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.seen_examples = seen_examples
        self.public_model = public_model
        self.public_grad_ex = public_grad_ex
        self.reconstructor = reconstructor
        self.n_sigma = n_sigma
        
        if dataset == 'Cifar10':
            self.input_shape = (3, 32, 32)
            self.num_classes = 10
            # trainset, testset, d_norm, d_denorm = datasets_Cifar10()
            _, testset, d_norm, self.d_denorm = datasets_Cifar10()
        elif dataset == 'Cifar100':
            self.input_shape = (3, 32, 32)
            self.num_classes = 100
            _, testset, d_norm, self.d_denorm = datasets_Cifar100()

        if loader_type == 'testloader':
            sampler = torch.utils.data.RandomSampler(testset, replacement=True, num_samples=int(1e9))
            self.dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=sampler)
        elif loader_type == 'randomloader':
            random_dataset = RandomImageDataset(num_samples=10000, input_shape=self.input_shape, num_classes=self.num_classes, d_norm=d_norm)
            self.dataloader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        self.reference_model, self.reference_grad_ex = load_clean(os.path.join('weights', dataset, 'standard_new_EP-1_init.params'), num_classes=self.num_classes)

    @classmethod
    def get_bdW(self, public_model, pub_loss_fn, grad_ex, X, y):
        batch_public_loss=pub_loss_fn(public_model(X), y)
        
        batch_public_gradients = torch.autograd.grad(
            outputs=batch_public_loss,
            inputs=public_model.parameters(),
            grad_outputs=torch.eye(batch_size, device=device),
            is_grads_batched=True,
            allow_unused=False
        )
        
        with torch.inference_mode():
            batch_dWflat=torch.cat([l.flatten(start_dim=1) for l in batch_public_gradients],dim=1)
            bdW = grad_ex.par_sel(batch_dWflat, single_grad=False, flat_cat=True)
            
        return batch_dWflat, bdW

    def get_norm_results(self, model, grad_ex, seen_examples=20):
        norms_log = []
        # psnrs = []
        # image_outs = []
        for i, (X, y) in tqdm(enumerate(self.dataloader), total=seen_examples, desc='Evaluating grad norms'):
            X, y = X.to(device), y.to(device)
            if X.shape != (batch_size, *self.input_shape):
                continue
            
            # scores_batch = property_scores(X, attack_prop)
            # order = torch.argsort(scores_batch, descending=True)
            # X, y, scores_batch = X[order], y[order], scores_batch[order]
            
            batch_dWflat, bdW = self.get_bdW(model, self.pub_loss_fn, grad_ex, X, y) 
            with torch.no_grad():
                bdW = torch.sum(bdW, dim=0)
                batch_dWflat_norm = torch.linalg.vector_norm(batch_dWflat, dim=1)
                
                # if self.reconstructor is not None:
                #     reconstructor_o = self.reconstructor(bdW).reshape(*self.input_shape)
                #     reconstructor_o = self.d_denorm(reconstructor_o).detach().cpu()
                #     metrics = run_metrics(reconstructor_o.unsqueeze(0), self.d_denorm(X))
                #     origin = self.d_denorm(X[metrics['selector']]).squeeze().detach().cpu()
                #     psnrs.append(metrics['max_psnr'])
                #     image_outs.append(reconstructor_o)
                #     image_outs.append(origin)
                
            if i >= seen_examples:
                break
            
            norms_log.append(batch_dWflat_norm.detach().cpu().numpy()) # (batch_size,)
        
        return norms_log
    
    def __call__(self, skip_reference=False):
        if not skip_reference:
            self.reference_norms_log = self.get_norm_results(self.reference_model, self.reference_grad_ex, self.seen_examples)
            self.reference_norm_cvs = list(map(cv, self.reference_norms_log))
            reference_norm_cv_mean, reference_norm_cv_std = np.mean(self.reference_norm_cvs), np.std(self.reference_norm_cvs)
            print(f'Reference Norms CV: {reference_norm_cv_mean} ± {reference_norm_cv_std}')
        else:
            reference_norm_cv_mean, reference_norm_cv_std = 0, 0
        
        self.public_norms_log = self.get_norm_results(self.public_model, self.public_grad_ex, self.seen_examples)
        self.public_norm_cvs = list(map(cv, self.public_norms_log))
        public_norm_cv_mean, public_norm_cv_std = np.mean(self.public_norm_cvs), np.std(self.public_norm_cvs)
        print(f'Public Model Norms CV: {public_norm_cv_mean} ± {public_norm_cv_std}')
        
        if public_norm_cv_mean >= reference_norm_cv_mean + self.n_sigma * reference_norm_cv_std:
            return True # Malicious
        else:
            return False # Clean
            

if __name__ == "__main__":
    results = []

    public_model, public_grad_ex, reconstructor = load_malicious(dataset, attack_prop, batch_size=batch_size, num_classes=num_classes, input_shape=input_shape)
    granite = GRANITE(dataset, public_model, public_grad_ex, reconstructor, loader_type=loader_type, seen_examples=seen_examples, n_sigma=3)
    granite()
    public_norm_cv_mean, public_norm_cv_std = np.mean(granite.public_norm_cvs), np.std(granite.public_norm_cvs)
    print(f'Public Malicious Model Norms CV (testloader): {public_norm_cv_mean} ± {public_norm_cv_std}')
    reference_norm_cv_mean, reference_norm_cv_std = np.mean(granite.reference_norm_cvs), np.std(granite.reference_norm_cvs)
    print(f'Reference Model Norms CV (testloader): {reference_norm_cv_mean} ± {reference_norm_cv_std}')

    ckpts = list(filter(lambda x: re.match(r'.*EP\d+.*\.params', x), os.listdir(os.path.join('weights', dataset))))
    ckpts = sorted(ckpts, key=lambda x: int(re.findall(r'EP(\d+)', x)[0]))

    for epoch, ckpt in enumerate(ckpts, start=1):
        print(f'Processing {ckpt}...')
        public_model, public_grad_ex = load_clean(os.path.join('weights', dataset, ckpt), num_classes=num_classes, seed=1337)
        reconstructor = None
        granite = GRANITE(dataset, public_model, public_grad_ex, reconstructor, loader_type=loader_type, seen_examples=seen_examples, n_sigma=3)
        prediction = granite(skip_reference=True)
        clean_public_norm_cv_mean, clean_public_norm_cv_std = np.mean(granite.public_norm_cvs), np.std(granite.public_norm_cvs)
        print(f'Public Clean Model Norms CV (testloader): {clean_public_norm_cv_mean} ± {clean_public_norm_cv_std}')
        
        results.append({
            "Epoch": epoch,
            "Model": "Clean Public Model",
            "Mean": clean_public_norm_cv_mean,
            "Std": clean_public_norm_cv_std,
        })
        
        results.append({
            "Epoch": epoch,
            "Model": "Malicious Public Model",
            "Mean": public_norm_cv_mean,
            "Std": public_norm_cv_std,
        })
        
        results.append({
            "Epoch": epoch,
            "Model": "Reference Model",
            "Mean": reference_norm_cv_mean,
            "Std": reference_norm_cv_std,
        })


    df = pd.DataFrame(results)
    df.to_csv(os.path.join('results', f'experiment_5_green_{loader_type}.csv'), index=False)
    
