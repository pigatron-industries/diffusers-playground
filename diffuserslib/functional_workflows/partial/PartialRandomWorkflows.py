from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.input import *


class RandomIntWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Random Integer", int, workflow=False, subworkflow=True)

    def build(self):
        min_max = IntTupleInputNode()
        random_int = RandomIntNode(min_max = min_max)
        return random_int
    

class RandomFloatWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Random Float", float, workflow=False, subworkflow=True)

    def build(self):
        min_max = FloatTupleInputNode()
        random_float = RandomFloatNode(min_max = min_max)
        return random_float


class RandomImageWorkflow(WorkflowBuilder):
    
    def __init__(self):
        super().__init__("Random Image", Image.Image, workflow=False, subworkflow=True)

    def build(self):
        paths = FileSelectInputNode()
        random_image = RandomImageNode(paths = paths)
        return random_image


class RandomImageUploadWorkflow(WorkflowBuilder):
    
    def __init__(self):
        super().__init__("Image Upload - Random", Image.Image, workflow=False, subworkflow=True)

    def build(self):
        images = ImageUploadInputNode()
        random_image = ListRandomNode(items = images)
        return random_image


class BatchImageUploadWorkflow(WorkflowBuilder):
    
    def __init__(self):
        super().__init__("Image Upload - Batch", Image.Image, workflow=False, subworkflow=True)

    def build(self):
        images = ImageUploadInputNode()
        image = ListCycleNode(items = images)
        return image