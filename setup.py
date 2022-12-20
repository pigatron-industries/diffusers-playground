import os, subprocess

def runcmd(cmd, shell=False):
    print(subprocess.run(cmd, stdout=subprocess.PIPE, shell=shell).stdout.decode('utf-8'))


def setup():
    runcmd(['mkdir workspace'], True)
    os.chdir("workspace")
    runcmd('pwd')
    runcmd(['pip', 'install', '-r', '../requirements.txt'])
    runcmd(['pip', 'install', '-e', 'git+https://github.com/joeyballentine/ESRGAN.git#egg=esrgan'])
    runcmd(['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip'])
    runcmd(['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/clip-interrogator.git#egg=clip-interrogator'])
    runcmd(['cp ../models/esrgan/* src/esrgan/models'], True)
    runcmd(['rm src/esrgan/input/*'], True)
    runcmd(['rm src/esrgan/output/*'], True)


setup()

