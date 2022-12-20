import os, subprocess
from PIL import Image

def chdirWorkspaceDirectory(subfolder):
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../workspace/src/" + subfolder))
    os.chdir(dir)


def runcmd(cmd, shell=False):
    print(subprocess.run(cmd, stdout=subprocess.PIPE, shell=shell).stdout.decode('utf-8'))


def upscaleESRGAN(inimage, scale=4, model="remacri"):
    chdirWorkspaceDirectory("esrgan")
    infile = "input/work.png"
    outfile = "output/work.png"
    model = f'4x_{model}.pth'
    inimage.save(infile)
    runcmd(['python', 'upscale.py', model])
    return Image.open(outfile)
