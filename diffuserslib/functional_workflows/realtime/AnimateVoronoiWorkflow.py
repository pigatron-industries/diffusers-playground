from diffuserslib.functional import *
from diffuserslib.functional_workflows.realtime.RealtimeVoronoiWorkflow import RealtimeVoronoiWorkflow

class AnimateVoronoiWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Animate Voronoi", Image.Image, workflow=True, subworkflow=False, realtime=False)


    def build(self):
        voronoi = RealtimeVoronoiWorkflow().build()
        
        num_frames_input = IntUserInputNode(value = 20, name = "num_frames")
        frame_aggregator = FrameAggregatorNode(frame = voronoi, num_frames = num_frames_input)
        frames_to_video = FramesToVideoNode(frames = frame_aggregator, fps = 10)
        return frames_to_video