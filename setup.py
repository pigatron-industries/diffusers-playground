import os, subprocess, sys

def runcmd(cmd, shell=False):
    print(subprocess.run(cmd, stdout=subprocess.PIPE, shell=shell).stdout.decode('utf-8'))


def setup(colab=False, clipInterrogator=False, diffuserScripts=False, rife=False, nafnet=False):
    runcmd(['mkdir -p workspace'], True)
    os.chdir("workspace")
    runcmd(['mkdir -p models'], True)
    runcmd('pwd')

    if(clipInterrogator):
        runcmd(['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip'])
        runcmd(['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/clip-interrogator.git#egg=clip-interrogator'])

    if(diffuserScripts):
        runcmd(['pip', 'install', '-e', 'git+https://github.com/huggingface/diffusers#egg=diffusers'])

    if(rife):
        runcmd(['pip', 'install', '-e', 'git+https://github.com/hzwer/Practical-RIFE.git#egg=rife'])

    if(nafnet):
        runcmd(['pip', 'install', '-e', 'git+https://github.com/megvii-research/NAFNet#egg=nafnet'])

    os.chdir("..")
    if(colab):
        runcmd(['pip', 'install', '-r', 'requirements_colab.txt'])
    else:
        runcmd(['pip', 'install', '-r', 'requirements.txt'])


if __name__ == "__main__":
    setup(colab=False, clipInterrogator=True, diffuserScripts=True, rife=True, nafnet=True)

