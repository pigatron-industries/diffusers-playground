from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from diffuserslib.functional.composites import *

name = "Generate Voronoi"

def build():
    return GenVoronoiWorkflowNode(size=(512, 512), points = uniform_distribution((50, 2)))


class GenVoronoiWorkflowNode(GenVoronoiNode):
    pass

