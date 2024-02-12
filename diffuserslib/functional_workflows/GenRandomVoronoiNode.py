from diffuserslib.functional import *

def name():
    return "Random Voronoi"

def build():
    size_user_input = SizeUserInputNode(value = (512, 512), name = "size")
    num_points_input = IntUserInputNode(value = 20, name = "num_points")

    random_points = RandomPoints2DNode(num_points = num_points_input)
    new_image = NewImageNode(size = size_user_input, background_colour = (0, 0, 0))
    voronoi = DrawVoronoiNode(image = new_image, points = random_points)

    return voronoi
