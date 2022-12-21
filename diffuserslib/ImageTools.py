import os, subprocess, sys
from PIL import Image

def chdirWorkspaceDirectory(subfolder):
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../workspace/src/" + subfolder))
    os.chdir(dir)


def addPath(subfolder):
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../workspace/src/" + subfolder))
    sys.path.append(dir)


def addToolPatha():
    addPath("esrgan")
    addPath("blip")
    addPath("clip-interrogator")


def runcmd(cmd, shell=False):
    print(subprocess.run(cmd, stdout=subprocess.PIPE, shell=shell).stdout.decode('utf-8'))


def upscaleEsrgan(inimage, scale=4, model="remacri", cpu=False):
    prevcwd = os.getcwd()
    chdirWorkspaceDirectory("esrgan")
    infile = "input/work.png"
    outfile = "output/work.png"
    model = f'4x_{model}.pth'
    inimage.save(infile)
    cmd = ['python', 'upscale.py', model]
    if(cpu):
        cmd.append('--cpu')
    runcmd(cmd)
    outimage = Image.open(outfile)
    os.chdir(prevcwd)
    return outimage


DEFAULT_CLIP_MODEL = "ViT-L-14/openai"

class ClipInterrogator():
    def __init__(model=DEFAULT_CLIP_MODEL):

