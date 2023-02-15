from .SceneDef import Scene, Sequence
from .Transforms import *
from ..FileUtils import getPathsFiles
from ..inference.DiffusersPipelines import DiffusersPipelines
from ..StringUtils import padNumber
from typing import List
from pathlib import Path
import sys


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class SceneRenderer():
    def __init__(self, inputfolder:str, ouputfolder:str, pipelines:DiffusersPipelines):
        self.ouputfolder = ouputfolder
        self.inputfolder = inputfolder
        self.pipelines = pipelines


    def renderSequenceFrames(self, sequence:Sequence):
        outdir = f"{self.ouputfolder}/{sequence.name}"
        Path(outdir).mkdir(parents=True, exist_ok=True)

        if(sequence.initimage is not None):
            currentframe = Image.open(f"{self.inputfolder}/{sequence.initimage}")
        else:
            currentframe = None

        for frame in range(0, sequence.length+1):
            for transform in sequence.transforms:
                currentframe = transform(currentframe)
            currentframe.save(f"{outdir}/{padNumber(frame, 5)}.png")


    def renderVideo(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f"{self.ouputfolder}/video.mp4", fourcc, 30.0, (512, 512))
        for scenepath, scene in getPathsFiles(f"{self.ouputfolder}/*/"):
            for framepath, framefile in getPathsFiles(f"{scenepath}/*.png"):
                image = cv2.imread(framepath)
                video.write(image)
        video.release()

