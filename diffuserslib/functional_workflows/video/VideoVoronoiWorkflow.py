from diffuserslib.functional import *
from diffuserslib.functional_workflows.video.FramesVoronoiWorkflow import FramesVoronoiWorkflow

class VideoVoronoiWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video - Voronoi", Image.Image, workflow=True, subworkflow=False, realtime=False)


    def build(self):
        voronoi_frames = FramesVoronoiWorkflow().build()
        frames_to_video = FramesToVideoNode(frames = voronoi_frames, fps = 7.5)
        return frames_to_video
