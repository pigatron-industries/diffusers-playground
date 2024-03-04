from diffuserslib.functional import *


class ImageRotateFeedbackWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Rotation", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        image_input = ImageSelectInputNode(name = "image")
        angle_input = FloatUserInputNode(value = 3.6, name = "angle")
        rotate = RotateImageNode(image = image_input, angle = angle_input)
        return rotate
    