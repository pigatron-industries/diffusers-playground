from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from diffuserslib.functional.elements.diffusers.ImageDiffusionNode import *
from PIL import Image


def build():
    return ImageDiffusionNode()

def name():
    return ImageDiffusionNode.name

