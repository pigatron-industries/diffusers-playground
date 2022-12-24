import os, subprocess, sys

def runcmd(cmd, shell=False):
    print(subprocess.run(cmd, stdout=subprocess.PIPE, shell=shell).stdout.decode('utf-8'))


def setup(colab=False):
    runcmd(['mkdir workspace'], True)
    os.chdir("workspace")
    runcmd('pwd')
    if(colab):
        runcmd(['pip', 'install', '-r', '../requirements_colab.txt'])
    else:
        runcmd(['pip', 'install', '-r', '../requirements.txt'])
    runcmd(['pip', 'install', '-e', 'git+https://github.com/joeyballentine/ESRGAN.git#egg=esrgan'])
    runcmd(['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip'])
    runcmd(['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/clip-interrogator.git#egg=clip-interrogator'])
    runcmd(['cp ../models/esrgan/* src/esrgan/models'], True)
    runcmd(['rm src/esrgan/input/*'], True)
    runcmd(['rm src/esrgan/output/*'], True)
    os.chdir("..")


if __name__ == "__main__":
    setup()

