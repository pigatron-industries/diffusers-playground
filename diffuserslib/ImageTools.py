import os 

def chdirWorkspaceDirectory(subfolder):
    filepath = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.normpath(os.path.join(filepath, "../workspace/src/" + subfolder))
    os.chdir(dir)


def upscaleESRGAN(inimage, scale):
    chdirWorkspaceDirectory("esrgan")
