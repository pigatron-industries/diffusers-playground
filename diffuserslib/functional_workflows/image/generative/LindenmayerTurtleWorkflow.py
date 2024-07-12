from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.WorkflowBuilder import *
from diffuserslib.functional.nodes.text.LindenmayerSystemNode import LindenmayerSystemNode
from diffuserslib.functional.nodes.image.generative.TurtleNode import TurtleNode
from diffuserslib.functional.nodes.image.generative.NewImageNode import NewImageNode


class LindenmayerTurtleWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Generation - Lindenmayer System Turtle", Image.Image, workflow=True, subworkflow=True, realtime=True)


    def build(self):
        size_input  = SizeUserInputNode(value = (512, 512), name = "size")
        iterations_input = IntUserInputNode(value = 5, name = "interations")
        axiom_input = StringUserInputNode(value = "F", name = "axiom")
        rules_input = TextAreaLinesInputNode(value = "F->F+F-F", name = "rules")
        init_position_input = FloatTupleInputNode(value = (0.5, 0.5), labels=("X", "Y"), name = "init_position")
        init_heading_input = FloatUserInputNode(value = 270, name = "init_heading")
        init_line_length_input = FloatUserInputNode(value = 10, name = "init_line_length")
        init_turning_angle_input = FloatUserInputNode(value = 90, name = "init_turning_angle")
        line_length_factor_input = FloatUserInputNode(value = 1.0, name = "line_length_factor")

        newimage = NewImageNode(name = "image", size = size_input, background_colour = "black")
        lsystem = LindenmayerSystemNode(axiom = axiom_input, rules = rules_input, iterations = iterations_input, name = "lsystem")
        turtle = TurtleNode(image = newimage, instructions = lsystem, colour = "white", name = "turtle", 
                            init_position = init_position_input, init_heading = init_heading_input,
                            init_turning_angle = init_turning_angle_input, init_line_length = init_line_length_input, 
                            line_length_factor=line_length_factor_input)
        return turtle
