from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types import Video, VideosFuncType, FloatFuncType

import tempfile
import cv2
import numpy as np


class VideosToVideoNode(FunctionalNode):
    def __init__(self,
                 videos:VideosFuncType,
                 fps:FloatFuncType|None = None,
                 name:str = "videos_to_video"):
        super().__init__(name)
        self.addParam("videos", videos, List[Video])
        self.addParam("fps", fps, float)


    def process(self, videos:List[Video], fps:float|None) -> Video:
        temp_file = tempfile.NamedTemporaryFile(suffix = ".mp4", delete = True)
        first_frame = videos[0].getFrame(0)
        height = first_frame.height
        width = first_frame.width
        if(fps is None):
            fps = videos[0].getFrameRate()
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))

        for video in videos:
            for frame_num in range(video.getFrameCount()):
                frame = video.getFrame(frame_num)
                if(frame is None):
                    break
                if(frame.size != (width, height)):
                    frame = frame.resize((width, height))
                np_array = np.array(frame)
                cv2_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
                out.write(cv2_image)

        out.release()
        print(temp_file.name)
        return Video(frame_rate = fps, file = temp_file)
