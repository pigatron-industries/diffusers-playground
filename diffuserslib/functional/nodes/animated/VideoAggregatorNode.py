from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *


class VideoAggregatorNode(FunctionalNode):
    def __init__(self, 
                 videos:VideosFuncType,
                 num_videos:IntFuncType,
                 name:str = "video_aggregator"):
        super().__init__(name)
        self.addInitParam("num_videos", num_videos, int)
        self.addParam("videos", videos, List[Video])
        self.args = {}
        self.videos = []
        self.videos_done = 0


    def init(self, num_videos:int):
        self.num_videos = num_videos


    def __call__(self) -> List[Video]:
        self.videos = []
        self.videos_done = 0
        self.video()
        self.videos_done += 1
        for i in range(1, self.num_videos):
            self.video()
            self.videos_done += 1
        return self.videos
    

    def video(self) -> List[Video]:
        if(self.stopping):
            raise WorkflowInterruptedException("Workflow interrupted")
        self.flush()
        self.args = self.evaluateParams()
        videos = self.args["videos"]
        self.videos.extend(videos)
        return videos
    

    def getProgress(self) -> WorkflowProgress|None:
        if len(self.videos) == 0:
            return WorkflowProgress(0, None)
        else:
            return WorkflowProgress(float(self.videos_done) / float(self.num_videos), self.videos)
