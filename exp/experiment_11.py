import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
import pickle
import torch
from copy import deepcopy

from exp.granite import GraniteConfig, GRANITE, load_clean, load_malicious

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = GraniteConfig(
        dataset='Cifar100',
        loader_type='testloader',
        batch_size=64,
        seen_examples=300, #300
        seed=42,
        par_sel_seed=98,
        par_sel_num=8400,
        par_sel_frac=0.001,
        num_classes=100,
        input_shape=(3, 32, 32),
        device=device
    )
    
    results = {}
    attack_prop = 'green'
    
    model, grad = load_clean(config=config)
    granite = GRANITE(model, grad, config=config)
    ref_norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_prop, enable_tqdm=True)
    print(f'Reference Model - Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    results['ref_norm_cvs'] = ref_norm_cvs
    
    ckpts = os.listdir('seer_weights/Cifar100/bs_64')
    ckpts = sorted(ckpts, key=lambda x: int(re.findall(r'.(\d+)', x)[0]))
    
    per_ckpt_results = {}
    for i, ckpt in enumerate(ckpts):
        print(f'Loading ckpt: {ckpt}')
        model, grad_ex, reconstructor = load_malicious(config=config, weight_path=f'seer_weights/Cifar100/bs_64/{ckpt}')
        granite = GRANITE(model, grad_ex, reconstructor, config=config)
        malicious_norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_prop, enable_tqdm=True)
        print(f'Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}, psnrs: {np.mean(psnrs)}')
        per_ckpt_results[i] = {
            'malicious_norm_cvs': malicious_norm_cvs,
            'psnrs': psnrs,
            'image_outs': granite.image_outs
        }
        
    results['per_ckpt_results'] = per_ckpt_results
    import pickle
    with open('results/experiment_11_results.pkl', 'wb') as f:
        pickle.dump(results, f)