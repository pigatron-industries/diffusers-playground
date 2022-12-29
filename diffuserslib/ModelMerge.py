import torch
import os, subprocess, sys


def chdirDiffuserScripts():
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../workspace/src/diffusers/scripts"))
    os.chdir(dir)


def mergeModels(file_path, model_a, model_b, alpha, fp16):
    model_0 = torch.load(f'{file_path}/{model_a}')
    model_1 = torch.load(f'{file_path}/{model_b}')
    theta_0 = model_0['state_dict']
    theta_1 = model_1['state_dict']

    filename = f'{model_a[:-5]}_{model_b[:-5]}_{alpha}.ckpt'

    print(f'Merging Common Weights...')
    for key in theta_0.keys():
        if 'model' in key and key in theta_1:
            theta_0[key] = (1 - alpha) * theta_0[key] + alpha * theta_1[key]
    print(' Done!')

    print(f'Merging Distinct Weights...')
    for key in theta_1.keys():
        if 'model' in key and key not in theta_0:
            theta_0[key] = theta_1[key]
    print(' Done!')

    if fp16:
        print(f'Converting to FP16...')
        for key in theta_0.keys():
            if 'model' in key and key not in theta_0:
                theta_0[key] = theta_0[key].to(torch.float16)
        print(' Done!')

    print(f'Saving Model...')
    torch.save(model_0, f'{file_path}/{filename}')
    print(' Done!')

    print(' ===============Merge Complete===============')
    return f'{file_path}/{filename}'