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
        seen_examples=50,
        seed=42,
        par_sel_seed=98,
        par_sel_num=8400,
        par_sel_frac=0.001,
        num_classes=100,
        input_shape=(3, 32, 32),
        device=device,
    )
    
    # reference
    model, grad_ex = load_clean(config=config)
    granite = GRANITE(model, grad_ex, config=config)
    ref_norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop='bright', enable_tqdm=True)
    print(f'Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    
    # clean
    model, grad_ex = load_clean(config=config)
    granite = GRANITE(model, grad_ex, config=config)
    clean_norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop='bright', enable_tqdm=True)
    print(f'Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    
    # malicious
    model, grad_ex, reconstructor = load_malicious(attack_prop='bright', config=config)
    granite = GRANITE(model, grad_ex, config=config)
    malicious_norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop='bright', enable_tqdm=True)
    print(f'Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    
    import pickle
    with open('results/experiment_1.pkl', 'wb') as f:
        pickle.dump({
            'ref_norm_cvs': ref_norm_cvs,
            'clean_norm_cvs': clean_norm_cvs,
            'malicious_norm_cvs': malicious_norm_cvs
        }, f)
