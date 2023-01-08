import torch
import os, subprocess, sys
import urllib.request as request
from urllib.parse import urlparse
from safetensors.torch import load_file


def chdirDiffuserScripts():
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../workspace/src/diffusers/scripts"))
    os.chdir(dir)


def getModelsDir():
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../workspace/models"))
    return dir


def chdirModels():
    os.chdir(getModelsDir())


def mergeModels(file_path, model_a, model_b, alpha, fp16):
    model_0 = torch.load(f'{file_path}/{model_a}')
    model_1 = torch.load(f'{file_path}/{model_b}')
    if "state_dict" in model_0:
        theta_0 = model_0['state_dict']
    else:
        theta_0 = model_0

    if "state_dict" in model_1:
        theta_1 = model_1['state_dict']
    else:
        theta_1 = model_1

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


def runcmd(cmd, shell=True):
    print(subprocess.run(cmd, stdout=subprocess.PIPE, shell=shell).stdout.decode('utf-8'))


def convertToDiffusers(modelname, modelsdir=None):
    if(modelsdir is None):
        modelsdir = getModelsDir()
    chdirDiffuserScripts()
    modelpath = getModelsDir() + "/" + modelname + ".ckpt"
    dumpFolder = modelpath[:-5]
    print(f"Converting model to Diffusers")
    print(f'python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path {modelpath} --dump_path {dumpFolder} --extract_ema')
    runcmd([f'python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path {modelpath} --dump_path {dumpFolder} --extract_ema'])


def convertSafetensorsToCkpt(modelname, modelsdir=None):
    if(modelsdir is None):
        modelsdir = getModelsDir()
    fromfile = modelsdir + "/" + modelname + ".safetensors"
    tofile = modelsdir + "/" + modelname + ".ckpt"
    weights = load_file(fromfile, device='cpu')
    with open(tofile, "wb") as f:
        torch.save(weights, f)


def downloadModel(url, modelname):
    print("Downloading model from " + url)
    chdirModels()
    filename =  modelname+".ckpt"
    opener = request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    request.install_opener(opener)
    request.urlretrieve(url, filename)
