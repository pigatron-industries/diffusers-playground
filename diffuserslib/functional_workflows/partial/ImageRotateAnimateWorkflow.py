from diffuserslib.functional import *


class ImageRotateAnimateWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Rotation - Animate", Image.Image, workflow=True, subworkflow=True, realtime=True)


    def build(self):
        time_delta_user_input = FloatUserInputNode(value = 0.01, name = "time_delta")
        image_input = ImageSelectInputNode(name = "image")

        angle_animate = AnimateFloatNode(interpolator=LinearInterpolation(0, 360), dt=time_delta_user_input)
        
        rotate = RotateImageNode(image = image_input, angle = angle_animate)
        return rotate