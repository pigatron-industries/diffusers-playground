from nicegui import app
from threading import Thread
from diffuserslib.inference import GenerationParameters, DiffusersPipelines, TiledGenerationParameters, UpscaleGenerationParameters
from diffuserslib.ImageUtils import base64EncodeImage, alphaToMask
from diffuserslib.inference.DiffusersUtils import tiledProcessorCentred, tiledImageToImageMultipass, tiledGeneration, tiledInpaint, compositedInpaint
from diffuserslib.processing.ProcessingPipelineFactory import ProcessingPipelineBuilder
from diffuserslib import ImageTools
from diffuserslib.processing.processors.transformers import *
from diffuserslib.processing.processors.filters import *
from diffuserslib.functional.nodes.image.diffusers.TileSizeCalculatorNode import TileSizeCalculatorNode
from diffuserslib.functional.nodes.image.diffusers.ImageDiffusionTiledNode import ImageDiffusionTiledNode
from diffuserslib.functional.nodes.image.generative.TileMaskNode import TileMaskNode
from diffuserslib.inference.GenerationParameters import ControlImageParameters, ControlImageType
from .Clipboard import ClipboardContentDTO, Clipboard
from typing import List, Tuple
from PIL import Image
import sys


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class DiffusersJob():
    def __init__(self):
        self.thread:Thread|None = None
        self.status = { "status":"none", "action":"none", "description": "", "total": 0, "done": 0}


class RestApi:

    job: DiffusersJob = DiffusersJob()
    tools: ImageTools = ImageTools(device = 'mps') #TODO get device from global config, convert to Upscale workflow


    @staticmethod
    @app.get("/api/")
    def info():
        return 'stable-diffusion'
    

    @staticmethod
    @app.post("/api/clipboard")
    def writeClipboard(clipboard:ClipboardContentDTO):
        Clipboard.writeDTO(clipboard)


    @staticmethod
    @app.get("/api/clipboard")
    def readClipboard():
        return Clipboard.readDTO()
    

    @staticmethod
    @app.get("/api/models")
    def models(type:str, base:str|None = None):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("Pipelines not initialized")
        if(type == "upscale"):
            presets = DiffusersPipelines.pipelines.presets.getModelsByType("upscale")
        else:
            presets = DiffusersPipelines.pipelines.presets.getModelsByTypeAndBase(type, base)
        models = [model.toDict() for model in presets.values()]
        return models
    

    @staticmethod
    @app.get("/api/loras")
    def loras(model:str):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("Pipelines not initialized")
        loranames = DiffusersPipelines.pipelines.getLORAList(model)
        return loranames
    
    
    @staticmethod
    @app.get("/api/async")
    def getJobAsync():
        return RestApi.job.status


    @staticmethod
    def startAsync(action, function, params):
        if (RestApi.job.thread is not None and RestApi.job.thread.is_alive()):
            return RestApi.getJobAsync()
        RestApi.job.thread = Thread(target = function, args=[params])
        RestApi.job.thread.start()
        RestApi.job.status = {"status":"running", "action": action}
        return RestApi.job.status


    @staticmethod
    def updateProgress(description, total, done):
        RestApi.job.status['description'] = description
        RestApi.job.status['total'] = total
        RestApi.job.status['done'] = done


    @staticmethod
    @app.post("/api/async/generate")
    def generateAsync(params:GenerationParameters):
        return RestApi.startAsync("generate", RestApi.generateRun, params)


    @staticmethod
    @app.post("/api/generate")
    def generateRun(params:GenerationParameters):
        RestApi.validateParams(params)

        # print(params)
        try:
            print('=== generate ===')
            RestApi.prescaleBefore(params)

            outputimages = []
            for i in range(0, params.batch):
                RestApi.updateProgress(f"Running", params.batch, i)
                outimage, usedseed = DiffusersPipelines.pipelines.generate(params)
                outimage = RestApi.prescaleAfter([outimage], params)[0]
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            RestApi.job.status = { "status":"finished", "action":"generate", "images": outputimages }
            return RestApi.job.status

        except Exception as e:
            RestApi.job.status = { "status":"error", "action":"generate", "error":str(e) }
            raise e
        

    @staticmethod
    @app.post("/api/async/inpaint")
    def inpaintAsync(params:GenerationParameters):
        return RestApi.startAsync("inpaint", RestApi.inpaintRun, params)


    @staticmethod
    @app.post("/api/inpaint")
    def inpaintRun(params:GenerationParameters):
        RestApi.validateParams(params)
        
        try:
            print('=== inpaint ===')
            initimageparams = params.getInitImage()
            if (initimageparams is None):
                raise Exception("No init image provided")
            if (params.getMaskImage() is None):
                maskimage = alphaToMask(initimageparams.image)
                params.setMaskImage(maskimage)

            RestApi.prescaleBefore(params)

            outputimages = []
            for i in range(0, params.batch):
                RestApi.updateProgress(f"Running", params.batch, i)

                outimage, usedseed = compositedInpaint(DiffusersPipelines.pipelines, params)

                outimage = RestApi.prescaleAfter([outimage], params)[0]

                # outimage = applyColourCorrection(initimage, outimage)

                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            RestApi.job.status = { "status":"finished", "action":"generate", "images": outputimages }
            return RestApi.job.status

        except Exception as e:
            RestApi.job.status = { "status":"error", "action":"generate", "error":str(e) }
            raise e


    @staticmethod
    @app.post("/api/async/generateTiled")
    def generateTiledAsync(params:TiledGenerationParameters):
        return RestApi.startAsync("generateTiled", RestApi.generateTiledRun, params)


    @staticmethod
    @app.post("/api/generateTiled")
    def generateTiledRun(params:TiledGenerationParameters):
        RestApi.validateParams(params)

        try:
            print('=== generateTiled ===')
            outputimages = []
            for i in range(0, params.batch):
                RestApi.updateProgress(f"Running", params.batch, i)
                if (params.tilemethod=="auto"):
                    # TODO make this use a full workflow, will need to convert params into conditioning param nodes
                    # TODO support use of differential img2img by passing a mask tile
                    tilesize_calc = TileSizeCalculatorNode(image_size = (params.width, params.height), overlap = params.tileoverlap)()
                    outimage = ImageDiffusionTiledNode.tiledGeneration(params=params, tilewidth=tilesize_calc[0], tileheight=tilesize_calc[1], overlap=params.tileoverlap)
                elif (params.tilemethod=="singlepass"):
                    outimage, usedseed = tiledProcessorCentred(tileprocessor=tiledGeneration, pipelines=DiffusersPipelines.pipelines, params=params, tilewidth=tilesize_calc[0], tileheight=tilesize_calc[1], 
                                                               overlap=params.tileoverlap, alignmentx=params.tilealignmentx, alignmenty=params.tilealignmenty)
                elif (params.tilemethod=="multipass"):
                    outimage, usedseed = tiledImageToImageMultipass(tileprocessor=tiledGeneration, pipelines=DiffusersPipelines.pipelines, params=params, tilewidth=params.tilewidth, tileheight=params.tileheight, 
                                                                    overlap=params.tileoverlap, passes=2, strengthMult=0.5)
                elif (params.tilemethod=="inpaint"):
                    outimage, usedseed = tiledProcessorCentred(tileprocessor=tiledInpaint, pipelines=DiffusersPipelines.pipelines, params=params, tilewidth=params.tilewidth, tileheight=params.tileheight, 
                                                               overlap=params.tileoverlap)
                else:
                    raise Exception(f"Unknown method: {params.tilemethod}")
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            RestApi.job.status = { "status":"finished", "action":"img2imgTiled", "images": outputimages }
            return RestApi.job.status

        except Exception as e:
            RestApi.job.status = { "status":"error", "action":"img2imgTiled", "error":str(e) }
            raise e


    @staticmethod
    @app.post("/api/async/upscale")
    def upscaleAsync(params:UpscaleGenerationParameters):
        return RestApi.startAsync("upscale", RestApi.upscaleRun, params)


    @staticmethod
    @app.post("/api/upscale")
    def upscaleRun(params:UpscaleGenerationParameters):
        params.generationtype = "upscale"
        RestApi.validateParams(params)
        # print(params)
        try:
            print('=== upscale ===')
            initimageparams = params.getInitImage()
            if (initimageparams is None):
                raise Exception("No init image provided")

            outputimages = []
            for i in range(0, params.batch):
                RestApi.updateProgress(f"Running", params.batch, i)
                if(params.upscalemethod == "diffusers"):
                    outimage, seed = DiffusersPipelines.pipelines.generate(params)
                elif(params.upscalemethod == "esrgan"):
                    outimage = RestApi.tools.upscaleEsrgan(initimageparams.image, scale=params.upscaleamount, model=params.models[0].name)
                else:
                    outimage = RestApi.tools.upscaleEsrgan(initimageparams.image, scale=params.upscaleamount)
                outputimages.append({ "image": base64EncodeImage(outimage) })

            RestApi.job.status = { "status":"finished", "action":"upscale", "images": outputimages }
            return RestApi.job.status

        except Exception as e:
            RestApi.job.status = { "status":"error", "action":"upscale", "error":str(e) }
            raise e
        

    @staticmethod
    @app.post("/api/async/preprocess")
    def preprocessAsync(params:GenerationParameters):
        return RestApi.startAsync("preprocess", RestApi.preprocessRun, params)


    @staticmethod
    @app.post("/api/preprocess")
    def preprocessRun(params:GenerationParameters):
        # print(params)
        try:
            print('=== preprocess ===')
            processor = str_to_class(params.models[0].name + 'Processor')()
            
            initimageparams = params.getInitImage()
            if (initimageparams is None):
                raise Exception("No init image provided")
            
            pipeline = ProcessingPipelineBuilder.fromImage(initimageparams.image)
            pipeline.addTask(processor)
            outimage = pipeline()

            RestApi.job.status = { "status":"finished", "action":"preprocess", "images": [{"image":base64EncodeImage(outimage)}] }
            return RestApi.job.status

        except Exception as e:
            RestApi.job.status = { "status":"error", "action":"upscale", "error":str(e) }
            raise e
        

    @staticmethod
    def validateParams(params:GenerationParameters):
        if (len(params.models) == 0):
            raise Exception("Must provider at least one model")


    @staticmethod
    def prescaleBefore(params:GenerationParameters):
        if (float(params.prescale) > 1):
            for controlimage in params.controlimages:
                controlimage.image = RestApi.tools.upscaleEsrgan(controlimage.image, int(params.prescale), "4x_remacri")
        elif (float(params.prescale) < 1):
            for controlimage in params.controlimages:
                controlimage.image = controlimage.image.resize((int(controlimage.image.width * float(params.prescale)), int(controlimage.image.height * float(params.prescale))), Image.LANCZOS)
        

    @staticmethod
    def prescaleAfter(images:List[Image.Image], params:GenerationParameters) -> List[Image.Image]:
        if (params.prescale > 1):
            prescaledimages = []
            for image in images:
                image = image.resize((int(image.width / params.prescale), int(image.height / params.prescale)), Image.LANCZOS)
                prescaledimages.append(image)
            return prescaledimages
        elif (params.prescale < 1):
            prescaledimages = []
            for image in images:
                image = RestApi.tools.upscaleEsrgan(image, int(1 / params.prescale), "4x_remacri")
                prescaledimages.append(image)
            return prescaledimages
        else:
            return images