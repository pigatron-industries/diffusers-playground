from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.Video import *
import cv2


class VideoFrameSplitterNode(FunctionalNode):
    def __init__(self, 
                 video:VideoFuncType,
                 name:str = "frame_splitter"):
        super().__init__(name)
        self.addInitParam("video", video, Video)
        self.framenum = 0
        self.framecount = 0
        self.capture = None
    

    def init(self, video:Video):
        self.video = video
        self.framenum = 0
        if(self.capture is not None):
            self.capture.release()
        self.capture = cv2.VideoCapture(video.file.name)
        self.framecount = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))


    def process(self) -> Image.Image|None:
        assert self.capture is not None
        if(self.framenum < self.framecount):
            ret, frame = self.capture.read()
            if not ret:
                raise ValueError("Could not read video file")
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.framenum += 1
            return frame
        else:
            return None
