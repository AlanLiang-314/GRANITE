import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
import pickle
from copy import deepcopy

from exp.granite import GraniteConfig, GRANITE, load_clean, load_malicious

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--attack_prop', type=str, default='green', help='Attack property to evaluate')
args = argparser.parse_args()


if __name__ == '__main__':

    config = GraniteConfig(
        dataset='Cifar100',
        loader_type='testloader',
        batch_size=64,
        seen_examples=100,
        seed=42,
        par_sel_seed=98,
        par_sel_num=8400,
        par_sel_frac=0.001,
        num_classes=100,
        input_shape=(3, 32, 32),
        device='cuda:0'
    )
    
    results = {}
    # reference
    attack_prop = args.attack_prop
    
    config.loader_type = 'testloader'
    model, grad_ex = load_clean(config=config)
    granite = GRANITE(model, grad_ex, config=config)
    norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_prop)
    print(f'Ref Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    results['granite_ref_cvs_ep0'] = norm_cvs
    
    model, grad_ex = load_clean(weight_path='weights/Cifar100/reference_model_class100.params', config=config)
    granite = GRANITE(model, grad_ex, config=config)
    norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_prop)
    print(f'Ref Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    results['granite_ref_cvs'] = norm_cvs

    
    # malicious 
    model, grad_ex, reconstructor = load_malicious(attack_prop=attack_prop, config=config)
    granite = GRANITE(model, grad_ex, config=config)
    norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_prop)
    print(f'Malicious Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    results['granite_malicious_cvs'] = norm_cvs
    
    ckpts = list(filter(lambda x: re.match(r'.*EP\d+.*\.params', x), os.listdir(os.path.join('weights', config.dataset))))
    ckpts = sorted(ckpts, key=lambda x: int(re.findall(r'EP(\d+)', x)[0]))
    ckpts =  ["standard_new_EP-1_init.params"] + ckpts
    
    results['granite_clean_model'] = {}
    
    for i, ckpt in enumerate(ckpts):
        model, grad_ex = load_clean(os.path.join('weights', config.dataset, ckpt), config=config)
        granite = GRANITE(model, grad_ex, config=config)
        norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_prop)
        print(f'Clean Model {i} {ckpt} Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
        results['granite_clean_model'][i] = norm_cvs
        
    
    config.loader_type = 'randomloader'
    model, grad_ex = load_clean(config=config)
    granite = GRANITE(model, grad_ex, config=config)
    norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_prop)
    print(f'Ref Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    results['granite_plus_ref_cvs_ep0'] = norm_cvs
    
    model, grad_ex = load_clean(weight_path='weights/Cifar100/reference_model_class100.params', config=config)
    granite = GRANITE(model, grad_ex, config=config)
    norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_prop)
    print(f'Ref Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    results['granite_plus_ref_cvs'] = norm_cvs

    
    # malicious 
    model, grad_ex, reconstructor = load_malicious(attack_prop=attack_prop, config=config)
    granite = GRANITE(model, grad_ex, config=config)
    norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_prop)
    print(f'Malicious Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    results['granite_plus_malicious_cvs'] = norm_cvs
    
    ckpts = list(filter(lambda x: re.match(r'.*EP\d+.*\.params', x), os.listdir(os.path.join('weights', config.dataset))))
    ckpts = sorted(ckpts, key=lambda x: int(re.findall(r'EP(\d+)', x)[0]))
    ckpts =  ["standard_new_EP-1_init.params"] + ckpts
    
    results['granite_plus_clean_model'] = {}
    
    for i, ckpt in enumerate(ckpts):
        model, grad_ex = load_clean(os.path.join('weights', config.dataset, ckpt), config=config)
        granite = GRANITE(model, grad_ex, config=config)
        norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_prop)
        print(f'Clean Model {i} {ckpt} Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
        results['granite_plus_clean_model'][i] = norm_cvs
        
    with open(f'results/experiment_9_results_{attack_prop}.pkl', 'wb') as f:
        pickle.dump(results, f)