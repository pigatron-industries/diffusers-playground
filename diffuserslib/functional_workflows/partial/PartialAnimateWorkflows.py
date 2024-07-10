from diffuserslib.functional import *
from diffuserslib.functional.nodes.animated.AnimateFloatNode import *
from diffuserslib.functional.nodes.user import FloatTupleInputNode, FloatUserInputNode


class AnimateFloatRampWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Animate Float - Ramp", float, workflow=False, subworkflow=True)

    def build(self):
        dt_input = FloatUserInputNode(name="dt", value=0.01, format='%.4f')
        min_max_input = FloatTupleInputNode(name="min_max", value=(0.1, 0.2))
        random_float = AnimateFloatRampNode(dt=dt_input, min_max=min_max_input)
        return random_float
    

class AnimateFloatSineWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Animate Float - Sine", float, workflow=False, subworkflow=True)

    def build(self):
        dt_input = FloatUserInputNode(name="dt", value=0.01, format='%.4f')
        min_max_input = FloatTupleInputNode(name="min_max", value=(0.0, 1.0))
        random_float = AnimateFloatSineNode(dt=dt_input, min_max=min_max_input)
        return random_float