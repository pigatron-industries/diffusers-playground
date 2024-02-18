from diffuserslib.functional import *

class TestWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Test", Image.Image, workflow=True, subworkflow=False)


    def build(self):
        random_image = RandomImageNode(paths = FileSelectInputNode(name = "file_select"), name = "random_image")
        return random_image
