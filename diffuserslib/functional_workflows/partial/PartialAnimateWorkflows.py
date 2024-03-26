from diffuserslib.functional import *
from diffuserslib.functional.nodes.animated.AnimateFloatNode import *
from diffuserslib.functional.nodes.user import FloatTupleInputNode


class AnimateFloatWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Ramp Float", float, workflow=False, subworkflow=True)

    def build(self):
        dt_input = FloatUserInputNode()
        min_max_input = FloatTupleInputNode()
        random_float = RampFloatNode(dt=dt_input, min_max=min_max_input)
        return random_float