from diffuserslib.functional_workflows.video.FramesVoronoiWorkflow import FramesVoronoiWorkflow
from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.types import Video
from diffuserslib.functional.nodes.animated import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.video.diffusers import *
from diffuserslib.functional.nodes.user import *


class VideoVoronoiWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video - Voronoi", Video, workflow=True, subworkflow=False, realtime=False)


    def build(self):
        voronoi_frames = FramesVoronoiWorkflow().build()
        frames_to_video = FramesToVideoNode(frames = voronoi_frames, fps = 7.5)
        return frames_to_video
