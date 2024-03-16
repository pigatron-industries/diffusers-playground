from diffuserslib.functional import *


class LogarithmicSpiralWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Generation - Logarithmic Spiral", Image.Image, workflow=True, subworkflow=True)


    def build(self):
        size_user_input = SizeUserInputNode(value = (512, 512), name = "size")
        revolutions_input = FloatUserInputNode(value = 8, name = "revolutions")
        segment_angle_input = IntUserInputNode(value = 1, name = "segment_agnle")
        tightness_input = FloatUserInputNode(value = 0.25, format = '%.3f', name = "tightness")
        scale_input = FloatUserInputNode(value = 1.0, name = "scale")
        rotate_input = IntUserInputNode(value = 0, name = "rotate")

        new_image = NewImageNode(size = size_user_input, background_colour = (0, 0, 0))

        spiral = DrawLogarithmicSpiralNode(image = new_image, 
                                           revolutions = revolutions_input, 
                                           segment_angle = segment_angle_input, 
                                           tightness=tightness_input, 
                                           scale=scale_input,
                                           rotate=rotate_input)
        return spiral
    