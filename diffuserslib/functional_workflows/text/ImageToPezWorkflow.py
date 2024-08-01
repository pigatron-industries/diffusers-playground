from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.ImageToPezNode import ImageToPezNode


class ImageToPezWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Text Generation - Image to Pez", str, workflow=True, subworkflow=True)


    def build(self):
        image_input = ImageUploadInputNode(name="image_input")
        model_input = ListSelectUserInputNode(value="sdxl", options=list(ImageToPezNode.CLIP_MODELS.keys()), name="model_input")
        iterations_input = IntUserInputNode(value=100, name="iterations_input")

        pez = ImageToPezNode(image=image_input, clip_model=model_input, iterations=iterations_input, name="lsystem")
        return pez
