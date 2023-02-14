from .SceneDef import Scene, Sequence
from .Transforms import *
from .TransformsDiffusion import *
from ..FileUtils import getPathsFiles
from ..inference.DiffusersPipelines import DiffusersPipelines
from ..StringUtils import padNumber
from typing import List
from pathlib import Path
import sys

from IPython.display import display


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class SceneRenderer():
    def __init__(self, ouputfolder:str, pipelines:DiffusersPipelines):
        self.ouputfolder = ouputfolder
        self.pipelines = pipelines


    def renderSequenceFrames(self, sequence:Sequence, initimage):
        outdir = f"{self.ouputfolder}/{sequence.name}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        currentframe = initimage
        for frame in range(0, sequence.length):
            for transform in sequence.transforms:
                currentframe = transform(currentframe)
            currentframe.save(f"{outdir}/{padNumber(frame+1, 5)}.png")


    def renderVideo(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f"{self.ouputfolder}/video.mp4", fourcc, 30.0, (512, 512))
        for scenepath, scene in getPathsFiles(f"{self.ouputfolder}/*/"):
            for framepath, framefile in getPathsFiles(f"{scenepath}/*.png"):
                image = cv2.imread(framepath)
                video.write(image)
        video.release()

                
    def runDiffusion(self, scene: Scene, currentframe):
        if(scene.diffusion.type == "img2img"):
            nextframe, seed = self.pipelines.imageToImage(inimage=currentframe, 
                                        prompt=scene.diffusion.prompt, 
                                        negprompt=scene.diffusion.negprompt, 
                                        strength=scene.diffusion.strength, 
                                        scale=scene.diffusion.cfgscale, 
                                        scheduler=scene.diffusion.scheduler,
                                        seed=None)
        else:
            nextframe = currentframe
        return nextframe


