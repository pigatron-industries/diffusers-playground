from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types import *
import cv2


class VideoFrameSplitterNode(FunctionalNode):
    def __init__(self, 
                 video:VideoFuncType,
                 start_frame:IntFuncType = 0,
                 skip_frames:IntFuncType = 0,
                 name:str = "frame_splitter"):
        super().__init__(name)
        self.addInitParam("video", video, Video)
        self.addInitParam("start_frame", start_frame, int)
        self.addInitParam("skip_frames", skip_frames, int)
        self.framenum = 0
        self.framecount = 0
        self.capture = None
    

    def init(self, video:Video, start_frame:int = 0, skip_frames:int = 0):
        self.video = video
        self.skip_frames = skip_frames
        self.framenum = start_frame
        if(self.capture is not None):
            self.capture.release()
        self.capture = cv2.VideoCapture(video.file.name)
        self.framecount = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.framenum)


    def process(self) -> Image.Image|None:
        assert self.capture is not None
        if(self.framenum < self.framecount):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.framenum)
            ret, frame = self.capture.read()
            if not ret:
                raise ValueError("Could not read video file")
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.framenum += (self.skip_frames + 1)
            return frame
        else:
            return None
