from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.ImageToPezNode import ImageToPezNode


class LindenmayerSystemWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Text Generation - Image to Pez", str, workflow=True, subworkflow=True)


    def build(self):
        image_input = ImageUploadInputNode(name="image_input")

        pez = ImageToPezNode(image=image_input, name="lsystem")
        return pez
