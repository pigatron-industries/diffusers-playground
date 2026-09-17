from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.types import Video
from diffuserslib.functional.nodes.animated import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.video.diffusers import *
from diffuserslib.functional.nodes.user import *


class VideoGenerationVideoAggregationWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Video Generation - Videos", Video, workflow=True, subworkflow=False)


    def build(self):
        fps_input = FloatUserInputNode(name = "fps", value = 30)
        num_videos_input = IntUserInputNode(name = "num_videos", value = 20)
        video_input = VideoUploadInputNode(name = "video_input")

        video_aggregator = VideoAggregatorNode(videos = video_input, num_videos = num_videos_input)
        videos_to_video = VideosToVideoNode(videos = video_aggregator, fps = fps_input)

        feedback_init_image = ImageUploadInputNode(mandatory = False, display = "Initial Image", name = "feedback_init_image")
        feedback_image = FeedbackNode(type = Image.Image, input = video_input, init_value = feedback_init_image, name = "feedback_image", display_name="Feedback Image - Previous Output")
        return videos_to_video, feedback_image