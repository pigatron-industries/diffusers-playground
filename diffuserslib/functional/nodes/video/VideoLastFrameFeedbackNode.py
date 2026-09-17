from PIL import Image

from diffuserslib.functional.nodes.animated.FeedbackNode import FeedbackNode


class VideoLastFrameFeedbackNode(FeedbackNode):
    """Feedback node that extracts the last frame of a previous Video output as an image.

    Mirrors ``FeedbackNode`` (which returns the raw previous output) but converts a
    ``Video`` into its final frame so it can be fed back into the next batch as an
    ``Image.Image``. Returns ``init_value`` when there is no previous output.
    """

    def __init__(self,
                 init_value,
                 input=None,
                 type=Image.Image,
                 name:str = "video_feedback",
                 display_name:str = "Video Feedback"):
        super().__init__(init_value=init_value, type=type, input=input, name=name, display_name=display_name)


    def process(self) -> Image.Image:
        if self.input is None:
            raise Exception("Feedback input not set")
        video = self.input.getPreviousOutput()
        if video is None:
            return self.init_value
        return video.getFrame(video.getFrameCount() - 1)
