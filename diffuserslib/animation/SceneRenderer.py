from .SceneDef import Scene
from .Transforms import *
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
        self.createTransform(scene)
        currentframe = scene.initimage
        for frame, time in enumerate(self.transform.frametimings_diff):
            currentframe = self.runTransform(currentframe=currentframe)
            currentframe = self.runDiffusion(scene=scene, currentframe=currentframe)
            currentframe.save(f"{outdir}/{padNumber(frame+1, 5)}.png")


    def renderVideo(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f"{self.ouputfolder}/video.mp4", fourcc, 30.0, (512, 512))
        for scenepath, scene in getPathsFiles(f"{self.ouputfolder}/*/"):
            for framepath, framefile in getPathsFiles(f"{scenepath}/*.png"):
                image = cv2.imread(framepath)
                video.write(image)
        video.release()


    def createTransform(self, scene: Scene):
        transformClass = str_to_class(f"{scene.transform.type}Transform")
        self.transform = transformClass.from_params(scene.length, scene.transform)


    def runTransform(self, currentframe):
        return self.transform(currentframe)

                
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


