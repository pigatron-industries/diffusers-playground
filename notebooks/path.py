import os, sys

def setPathLocalNotebook():
    os.chdir("../workspace")
    sys.path.append('..')
    sys.path.append('./src/blip')
    sys.path.append('./src/clip-interrogator')
