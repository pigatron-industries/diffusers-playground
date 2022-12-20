import os, subprocess

os.chdir("workspace")

def setup():
    install_cmds = [
        ['pip', 'install', '-r', '../requirements.txt'],
        ['pip', 'install', '-e', 'git+https://github.com/joeyballentine/ESRGAN.git#egg=esrgan'],
        ['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip'],
        ['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/clip-interrogator.git#egg=clip-interrogator']
    ]
    for cmd in install_cmds:
        print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))


setup()

