import os, subprocess, sys


def runcmd(cmd, shell=False):
    label = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n$ {label}")
    subprocess.run(cmd, shell=shell, check=True)


def setup(rife=False, kohya=False, minimax_h3_mac=False):
    runcmd(['mkdir -p workspace'], True)
    os.chdir("workspace")
    runcmd('pwd')

    if(rife):
        if not os.path.isdir("Practical-RIFE"):
            runcmd(['git', 'clone', 'https://github.com/hzwer/Practical-RIFE.git'])
    if(kohya):
        if not os.path.isdir("sd-scripts"):
            runcmd(['git', 'clone', 'https://github.com/kohya-ss/sd-scripts.git'])
    if(minimax_h3_mac):
        if not os.path.isdir("minimax-h3-mac"):
            # runcmd(['git', 'clone', 'https://github.com/Argus-AiTeam/minimax-h3-mac.git'])
            runcmd(['git', 'clone', 'https://github.com/HayasakaInori/minimax-h3-mac.git'])
        os.chdir("minimax-h3-mac")
        if not os.path.isdir(".venv"):
            runcmd([sys.executable, '-m', 'venv', '.venv'])
        venv_python = os.path.join('.venv', 'bin', 'python')
        runcmd([venv_python, '-m', 'pip', 'install', '--upgrade', 'pip'])
        runcmd([venv_python, '-m', 'pip', 'install', '-r', 'requirements.txt'])


if __name__ == "__main__":
    setup(rife=True, kohya=True, minimax_h3_mac=True)
