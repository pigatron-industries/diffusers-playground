from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.generative import *
from diffuserslib.functional.WorkflowBuilder import *


class GeometricSpiralWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Generation - Geometric Spiral", Image.Image, workflow=True, subworkflow=True)


    def build(self):
        size_user_input = SizeUserInputNode(value = (512, 512), name = "size")
        draw_options = BoolListUserInputNode(value = [True, True], labels=["spiral", "rectangles"], name = "draw_options")
        direction_input = EnumSelectUserInputNode(value = DrawGeometricSpiralNode.Direction.RIGHT, enum = DrawGeometricSpiralNode.Direction, name = "direction")
        turn_input = EnumSelectUserInputNode(value = DrawGeometricSpiralNode.Turn.CLOCKWISE, enum = DrawGeometricSpiralNode.Turn, name = "turn")
        ratio_input = FloatUserInputNode(value = GOLDEN_RATIO, name = "ratio")
        iterations_input = IntUserInputNode(value = 7, name = "iterations")

        new_image = NewImageNode(size = size_user_input, background_colour = (0, 0, 0))

        spiral = DrawGeometricSpiralNode(image = new_image, ratio = ratio_input, iterations = iterations_input, direction = direction_input, turn = turn_input, draw_options = draw_options)
        return spiral
