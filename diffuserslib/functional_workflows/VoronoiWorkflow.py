from diffuserslib.functional_nodes import *

name = "Voronoi"

def build():
    size = InputValueNode(name="size", type=Tuple[int, int], value=(768, 768))
    image = NewImageNode(size=size, background_colour=(0, 0, 0))
    voronoi = DrawVoronoiNode(image = image, 
                            points = uniform_distribution((50, 2)))