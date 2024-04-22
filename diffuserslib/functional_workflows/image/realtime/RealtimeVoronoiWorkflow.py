from diffuserslib.functional import *


class RealtimeVoronoiWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Animate Voronoi - Bouncing Points", Image.Image, workflow=False, subworkflow=True, realtime=True)


    def build(self):
        # num_frames_input = IntUserInputNode(value = 20, name = "num_frames")
        time_delta_user_input = FloatUserInputNode(value = 0.01, name = "time_delta")
        size_user_input = SizeUserInputNode(value = (512, 512), name = "size")
        num_points_input = IntUserInputNode(value = 20, name = "num_points")
        radius_input = IntUserInputNode(value = 2, name = "radius")
        line_probability_input = FloatUserInputNode(value = 1.0, name = "line_probability")
        draw_options = BoolListUserInputNode(value = [True, True, True], labels=["bounded", "unbounded", "points"], name = "draw_options")

        random_bodies = RandomMovingBodies2DNode(num_bodies = num_points_input)
        bouncing_points = BouncingPoints2DNode(init_bodies = random_bodies, dt = time_delta_user_input)

        new_image = NewImageNode(size = size_user_input, background_colour = (0, 0, 0))
        voronoi = DrawVoronoiNode(image = new_image, points = bouncing_points, radius = radius_input, 
                                line_probability = line_probability_input, draw_options = draw_options)
        return voronoi
