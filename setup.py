import os, subprocess, sys

def runcmd(cmd, shell=False):
    print(subprocess.run(cmd, stdout=subprocess.PIPE, shell=shell).stdout.decode('utf-8'))


def setup(colab=False, rife=False, kohya=False):
    runcmd(['mkdir -p workspace'], True)
    os.chdir("workspace")
    runcmd('pwd')

    if(rife):
        runcmd(['git', 'clone', 'https://github.com/hzwer/Practical-RIFE.git'])
    if(kohya):
        runcmd(['git', 'clone', 'https://github.com/kohya-ss/sd-scripts.git'])

    os.chdir("..")
    if(colab):
        runcmd(['pip', 'install', '-r', 'requirements_colab.txt'])
    else:
        runcmd(['pip', 'install', '-r', 'requirements.txt'])


if __name__ == "__main__":
    setup(colab=False, rife=True, kohya=True)

