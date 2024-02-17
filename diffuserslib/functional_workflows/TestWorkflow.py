from diffuserslib.functional import *

def name():
    return "Test"

def build():
    random_image = RandomImageNode(paths = FileSelectInputNode(name = "file_select"), name = "random_image")
    return random_image
