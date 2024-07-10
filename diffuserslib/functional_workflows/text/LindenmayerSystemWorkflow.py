from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.LindenmayerSystemNode import LindenmayerSystemNode


class LindenmayerSystemWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Text Generation - Lindenmayer System", str, workflow=True, subworkflow=True)


    def build(self):
        iterations_input = IntUserInputNode(value = 5, name = "interations_input")
        axiom_input = StringUserInputNode(value = "F", name = "axiom_input")
        rules_input = TextAreaLinesInputNode(value = "F->F+F-F", name = "rules_input")

        lsystem = LindenmayerSystemNode(axiom=axiom_input, rules=rules_input, iterations=iterations_input, name="lsystem")
        return lsystem
