from email.mime import image
from numpy import diff, size
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.diffusers.user import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.image.transform import UpscaleImageNode
from diffuserslib.functional.nodes.image.transform import ResizeImageNode
from diffuserslib.functional.nodes.user import *
from PIL import Image

class ImageProcessorUpscaleWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Processor - Upscale", Image.Image, workflow=True, subworkflow=True)


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        scale_input = IntUserInputNode(value = 4, name = "scale")
        model_input = DictSelectUserInputNode(name = "model", value = "4x_remacri", options = UpscaleImageNode.UPSCALE_MODELS)
        
        # upscale = UpscaleImageNode(image = image_input, scale = scale_input)
        upscale = UpscaleImageNode(image = image_input, scale = scale_input).addParamsDict(model_input)
        return upscale
