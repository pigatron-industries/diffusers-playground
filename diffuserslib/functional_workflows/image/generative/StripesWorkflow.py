from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.generative import *
from diffuserslib.functional.WorkflowBuilder import *


class StripesWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Generation - Stripes", Image.Image, workflow=True, subworkflow=True)


    def build(self):
        size_user_input = SizeUserInputNode(value = (512, 512), name = "size")
        stripe_width_input = IntUserInputNode(value = 64, name = "strip_width")
        stripe_angle_input = FloatUserInputNode(value = 45, name = "stripe_angle")
        colour1_input = StringUserInputNode(value = "black", name = "colour1")
        colour2_input = StringUserInputNode(value = "white", name = "colour2")

        new_image = NewImageNode(size = size_user_input, background_colour = (0, 0, 0))

        spiral = DrawStripesNode(image = new_image, stripewidth = stripe_width_input, stripeangle = stripe_angle_input, colour1 = colour1_input, colour2 = colour2_input)
        return spiral
