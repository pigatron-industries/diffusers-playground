import os, subprocess, sys
from PIL import Image
from pathlib import Path

DEFAULT_CLIP_MODEL = "ViT-L-14/openai"


def chdirWorkspaceDirectory(subfolder):
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../workspace/src/" + subfolder))
    os.chdir(dir)


def addToolPath(subfolder):
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../workspace/src/" + subfolder))
    sys.path.append(dir)


def runcmd(cmd, shell=False):
    print(subprocess.run(cmd, stdout=subprocess.PIPE, shell=shell).stdout.decode('utf-8'))



class ImageTools():
    def __init__(self):
        self.addToolPaths()
        self.clipInterrogator = None


    def addToolPaths(self):
        addToolPath("esrgan")
        addToolPath("blip")
        addToolPath("clip-interrogator")


    def upscaleEsrgan(self, inimage, scale=4, model="remacri", cpu=False):
        prevcwd = os.getcwd()
        chdirWorkspaceDirectory("esrgan")
        infile = "input/work.png"
        outfile = "output/work.png"
        model = f'4x_{model}.pth'
        inimage.save(infile)

        from upscale import Upscale
        upscale = Upscale(model = "4x_remacri.pth", input=Path("input"), output=Path("output"))
        upscale.run()

        outimage = Image.open(outfile)
        if (scale != 4):
            outimage = outimage.resize((inimage.width*scale, inimage.height*scale))
        os.chdir(prevcwd)
        return outimage


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