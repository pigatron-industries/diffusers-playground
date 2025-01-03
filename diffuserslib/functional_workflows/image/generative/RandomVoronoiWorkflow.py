from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.generative import *
from diffuserslib.functional.nodes.input.RandomPoints2DNode import RandomPoints2DNode


class RandomVoronoiWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Generation - Random Voronoi", Image.Image, workflow=True, subworkflow=True)


    def build(self):
        size_user_input = SizeUserInputNode(value = (512, 512), name = "size")
        radius_input = IntUserInputNode(value = 2, name = "radius")
        width_input = IntUserInputNode(value = 1, name = "width")
        line_probability_input = FloatUserInputNode(value = 1.0, name = "line_probability")
        draw_options = BoolListUserInputNode(value = [True, True], labels=["bounded", "unbounded"], name = "draw_options")
        line_colour_input = ColourPickerInputNode(name = "line_colour")
        point_colour_input = ColourPickerInputNode(name = "point_colour")
        palette_input = ColourPaletteInputNode(name = "fill_palette")

        random_points = RandomPoints2DNode()
        new_image = NewImageNode(size = size_user_input, background_colour = (0, 0, 0))
        voronoi = DrawVoronoiNode(image = new_image, points = random_points, radius = radius_input, width = width_input,
                                  line_probability = line_probability_input, draw_options = draw_options, 
                                  fill_palette = palette_input, line_colour = line_colour_input, point_colour = point_colour_input)

        return voronoi
