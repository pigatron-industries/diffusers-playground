from diffuserslib.functional import *

def name():
    return "Random Voronoi"

def build():
    size_user_input = SizeUserInputNode(value = (512, 512), name = "size")
    num_points_input = IntUserInputNode(value = 20, name = "num_points")
    radius_input = IntUserInputNode(value = 2, name = "radius")
    line_probability_input = FloatUserInputNode(value = 1.0, name = "line_probability")
    draw_options = BoolListUserInputNode(value = [True, True, True], labels=["bounded", "unbounded", "points"], name = "draw_options")

    random_points = RandomPoints2DNode(num_points = num_points_input)
    new_image = NewImageNode(size = size_user_input, background_colour = (0, 0, 0))
    voronoi = DrawVoronoiNode(image = new_image, points = random_points, radius = radius_input, 
                              line_probablity = line_probability_input, draw_options = draw_options)

    return voronoi
