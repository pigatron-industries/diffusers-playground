from diffuserslib.functional import *


class RandomPoints2DUniformWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Random 2D Points - Unform Distribution", List[Vector], workflow=False, subworkflow=True)

    def build(self):
        num_points_input = IntUserInputNode(value = 20, name = "num_points")
        random_points = RandomPoints2DUniformNode(num_points = num_points_input)
        return random_points
    

class RandomPoints2DEdgePowerWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Random 2D Points - Edge Power Distribution", List[Vector], workflow=False, subworkflow=True)

    def build(self):
        num_points_input = IntUserInputNode(value = 20, name = "num_points")
        power_x_input = FloatUserInputNode(value = 1.0, name = "power_x")
        power_y_input = FloatUserInputNode(value = 1.0, name = "power_y")
        random_points = RandomPoints2DEdgePowerNode(num_points = num_points_input, power_x = power_x_input, power_y = power_y_input)
        return random_points
    