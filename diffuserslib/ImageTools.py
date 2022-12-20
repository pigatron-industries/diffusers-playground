import os, subprocess
from PIL import Image

def chdirWorkspaceDirectory(subfolder):
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../workspace/src/" + subfolder))
    os.chdir(dir)


def runcmd(cmd, shell=False):
    print(subprocess.run(cmd, stdout=subprocess.PIPE, shell=shell).stdout.decode('utf-8'))


def upscaleESRGAN(inimage, scale=4, model="remacri", cpu=False):
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
