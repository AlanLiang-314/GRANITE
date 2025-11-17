import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
import pickle
from copy import deepcopy

from exp.granite import GraniteConfig, GRANITE, load_clean, load_malicious

import argparse

args = argparse.ArgumentParser()
args.add_argument('--attack_dataset', type=str, default='Cifar100', help='Dataset to use for detection')
args.add_argument('--client_dataset', type=str, default='Cifar100', help='Dataset to use for detection')
args.add_argument('--attack_prop', type=str, default='red', help='Attack property to use for detection')
args.add_argument('--loader_type', type=str, default='testloader', help='Loader type to use for detection')
args = args.parse_args()


if __name__ == '__main__':
    ##################################################################################################
    # GRANITE Detection on Cifar100 with Cifar100 Public Model

    config = GraniteConfig(
        dataset='Cifar100',
        loader_type='testloader',
        batch_size=64,
        seen_examples=20, #300
        seed=42,
        par_sel_seed=98,
        par_sel_num=8400,
        par_sel_frac=0.001,
        num_classes=100,
        input_shape=(3, 32, 32),
        device='cuda:1',
    )
    
    config.num_classes = 100 if args.attack_dataset == 'Cifar100' else 10
    
    config.dataset = args.client_dataset
    config.loader_type = args.loader_type
    attack_props = args.attack_prop
    
    temp_config = deepcopy(config)
    temp_config.dataset = args.attack_dataset
    
    # reference
    model, grad_ex = load_clean(config=temp_config, weight_path=f"weights/{args.attack_dataset}/reference_model_class{config.num_classes}.params")
    granite = GRANITE(model, grad_ex, config=config)
    ref_norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_props, enable_tqdm=True)
    print(f'Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    # clean
    model, grad_ex = load_clean(config=temp_config)
    granite = GRANITE(model, grad_ex, config=config)
    clean_norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_props, enable_tqdm=True)
    print(f'Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    # malicious 
    model, grad_ex, reconstructor = load_malicious(attack_prop=attack_props, config=temp_config)
    granite = GRANITE(model, grad_ex, reconstructor, config=config)
    malicious_norm_cvs, norm_cvs_mean, norm_cvs_std, psnrs = granite.get_norm_results(attack_prop=attack_props, enable_tqdm=True)
    print(f'Norm CVs Mean: {norm_cvs_mean}, Std: {norm_cvs_std}')
    
    import pickle
    with open(f"results/experiment_8-{args.attack_dataset}-{args.client_dataset}-{attack_props}-{args.loader_type}.pkl", 'wb') as f:
        results = {'ref_cvs': ref_norm_cvs, 'clean_cvs': clean_norm_cvs, 'malicious_cvs': malicious_norm_cvs, 'psnrs': psnrs}
        pickle.dump(results, f)