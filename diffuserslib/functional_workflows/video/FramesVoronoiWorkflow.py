from diffuserslib.functional import *
from diffuserslib.functional_workflows.realtime.RealtimeVoronoiWorkflow import RealtimeVoronoiWorkflow

class FramesVoronoiWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Frames - Voronoi", List[Image.Image], workflow=False, subworkflow=True, realtime=False)


    def build(self):
        voronoi = RealtimeVoronoiWorkflow().build()
        num_frames_input = IntUserInputNode(value = 20, name = "num_frames")
        frame_aggregator = FrameAggregatorNode(frame = voronoi, num_frames = num_frames_input)
        return frame_aggregator
