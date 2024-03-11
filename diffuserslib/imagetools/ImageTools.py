import os, subprocess, sys
from typing_extensions import deprecated
from PIL import Image
from pathlib import Path
from .ESRGAN_upscaler import ESRGANUpscaler

DEFAULT_CLIP_MODEL = "ViT-L-14/openai"


def chdirWorkspaceDirectory(subfolder):
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../../workspace/src/" + subfolder))
    os.chdir(dir)


def chdirRootDirectory(subfolder = ""):
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../../" + subfolder))
    os.chdir(dir)


def addToolPath(subfolder):
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../../workspace/src/" + subfolder))
    print(f"toolpath: {dir}")
    sys.path.append(dir)


class ImageTools():
    def __init__(self, device = 'cuda'):
        self.addToolPaths()
        self.device = device
        self.clipInterrogator = None


    def addToolPaths(self):
        addToolPath("blip")
        addToolPath("clip-interrogator")
        addToolPath("nafnet")


    def upscaleEsrgan(self, inimage, scale=4, model="4x_remacri", tilewidth=512+64, tileheight=512+64, overlap=64):
        chdirRootDirectory()
        upscaler = ESRGANUpscaler(f'models/esrgan/{model}.pth', device=self.device)
        outimage = upscaler.upscaleTiled(inimage, scale=scale, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap)
        return outimage


    def deblurr(self, inimage):
        prevcwd = os.getcwd()
        chdirWorkspaceDirectory("nafnet")
        infile = "input/work.png"
        outfile = "output/work.png"


    def loadClipInterrogator(self, model = DEFAULT_CLIP_MODEL):
        from clip_interrogator import Interrogator, Config
        config = Config()
        config.blip_num_beams = 64
        config.blip_offload = False
        config.clip_model_name = model
        if(model == "ViT-L-14/openai"):
            config.chunk_size = 2048
            config.flavor_intermediate_count = 2048
        else:
            config.chunk_size = 1024
            config.flavor_intermediate_count = 1024
        self.clipInterrogator = Interrogator(config)


    def clipInterrogate(self, inimage, mode = 'best'):
        if (mode == 'best'):
            return self.clipInterrogator.interrogate(inimage)
        elif mode == 'classic':
            return self.clipInterrogator.interrogate_classic(inimage)
        else:
            return self.clipInterrogator.interrogate_fast(inimage)