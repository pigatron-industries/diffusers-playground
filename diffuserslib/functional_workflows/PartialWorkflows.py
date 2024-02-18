from diffuserslib.functional import *


class RandomIntWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Random Integer", int, workflow=False, subworkflow=True)

    def build(self):
        min_max = MinMaxIntInputNode()
        random_int = RandomIntNode(min_max = min_max)
        return random_int
    

class RandomFloatWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Random Float", float, workflow=False, subworkflow=True)

    def build(self):
        min_max = MinMaxFloatInputNode()
        random_float = RandomFloatNode(min_max = min_max)
        return random_float


class RandomImageWorkflow(WorkflowBuilder):
    
    def __init__(self):
        super().__init__("Random Image", Image.Image, workflow=False, subworkflow=True)

    def build(self):
        paths = FileSelectInputNode()
        random_image = RandomImageNode(paths = paths)
        return random_image

