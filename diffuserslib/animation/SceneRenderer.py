from .SceneDef import Scene
from .Transforms import *
from .TransformsDiffusion import *
from ..FileUtils import getPathsFiles
from ..inference.DiffusersPipelines import DiffusersPipelines
from ..StringUtils import padNumber
from pathlib import Path
import sys

from IPython.display import display


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class SceneRenderer():
    def __init__(self, ouputfolder:str, pipelines:DiffusersPipelines):
        self.ouputfolder = ouputfolder
        self.pipelines = pipelines
        self.transform:Transform = None


    def renderFrames(self, scene:Scene):
        outdir = f"{self.ouputfolder}/{scene.name}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        self.createTransforms(scene)
        currentframe = scene.initimage
        for frame in range(0, scene.length):
            currentframe = self.runTransforms(currentframe=currentframe)
            currentframe.save(f"{outdir}/{padNumber(frame+1, 5)}.png")


    def renderVideo(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f"{self.ouputfolder}/video.mp4", fourcc, 30.0, (512, 512))
        for scenepath, scene in getPathsFiles(f"{self.ouputfolder}/*/"):
            for framepath, framefile in getPathsFiles(f"{scenepath}/*.png"):
                image = cv2.imread(framepath)
                video.write(image)
        video.release()


    def createTransforms(self, scene: Scene):
        self.transforms = []
        for transformparams in scene.transforms:
            transformClass = str_to_class(f"{transformparams.type}Transform")
            transformparams.params['pipelines'] = self.pipelines
            self.transforms.append(transformClass.from_params(scene.length, transformparams))


    def runTransforms(self, currentframe):
        for transform in self.transforms:
            currentframe = transform(currentframe)
        return currentframe

                
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


