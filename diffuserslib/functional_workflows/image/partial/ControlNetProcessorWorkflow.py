from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.image.process import ControlNetProcessorNode
from diffuserslib.functional.nodes.user import ImageUploadInputNode
from diffuserslib.functional.nodes.user import ListSelectUserInputNode
from PIL import Image


class ControlNetProcessorWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Process - ControlNet Processor", Image.Image, workflow=False, subworkflow=True, realtime=False)


    def build(self):
        processor_input = ListSelectUserInputNode(name = "processor", value="canny", options = ControlNetProcessorNode.PROCESSORS)
        image_input = ImageUploadInputNode(name = "image", multiple=False)
        controlnet_processor = ControlNetProcessorNode(image = image_input, processor = processor_input, name = "controlnet_processor")
        return controlnet_processor
    