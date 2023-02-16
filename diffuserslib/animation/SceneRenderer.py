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
    def __init__(self, pipelines:DiffusersPipelines):
        self.pipelines = pipelines


    def renderSceneFrames(self, scene:Scene):
        outputdir = f"{scene.inputdir}/output"
        self.renderSequenceFrames(scene.sequences[0], outputdir)
        self.renderVideo(outputdir)


    def renderSequenceFrames(self, sequence:Sequence, outputdir:str):
        seqoutdir = f"{outputdir}/{sequence.name}"
        Path(seqoutdir).mkdir(parents=True, exist_ok=True)

        currentframe = None # todo initialise with last frame of previous sequence by default
        for inittransform in sequence.init:
            currentframe = inittransform(currentframe)

        for frame in range(0, sequence.length+1):
            for transform in sequence.transforms:
                currentframe = transform(currentframe)
            currentframe.save(f"{seqoutdir}/{padNumber(frame, 5)}.png")


    def renderVideo(self, framesdir):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f"{framesdir}/video.mp4", fourcc, 30.0, (512, 512))
        for scenepath, scene in getPathsFiles(f"{framesdir}/*/"):
            for framepath, framefile in getPathsFiles(f"{scenepath}/*.png"):
                image = cv2.imread(framepath)
                video.write(image)
        video.release()

