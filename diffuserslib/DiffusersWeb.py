from flask import request, jsonify
from flask_classful import FlaskView, route
from threading import Thread
from typing import List
from .inference.DiffusersPipelines import *
from .inference.DiffusersUtils import tiledProcessorCentred, tiledImageToImageMultipass, tiledImageToImage, tiledInpaint, compositedInpaint
from .inference.GenerationParameters import GenerationParameters, TiledGenerationParameters, UpscaleGenerationParameters
from .ImageUtils import base64EncodeImage, base64DecodeImage, base64DecodeImages, alphaToMask, applyColourCorrection
from .imagetools.ImageTools import ImageTools
from .processing.processors.TransformerProcessors import *
from .processing.processors.FilterProcessors import *
from .processing.ProcessingPipelineFactory import ProcessingPipelineBuilder
import json

from IPython.display import display, clear_output

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class DiffusersJob():
    def __init__(self):
        self.thread:Thread|None = None
        self.status = { "status":"none", "action":"none", "description": "", "total": 0, "done": 0}


class DiffusersView(FlaskView):
    route_base = '/'
    pipelines: DiffusersPipelines = None # type: ignore
    tools: ImageTools = None # type: ignore
    job: DiffusersJob = DiffusersJob()

    def __init__(self):
        pass


    @route("/api/", methods=["GET"])
    def info(self):
        return 'stable-diffusion'


    @route("/api/models", methods=["GET"])
    def models(self):
        if("type" in request.args):
            if(request.args["type"] == "inpaint"):
                presets = self.pipelines.presets.getModelsByType("inpaint")
            elif(request.args["type"] == "control"):
                presets = self.pipelines.presets.getModelsByType("controlnet")
            else:
                presets = self.pipelines.presets.getModelsByType("txt2img")
        models = [model.toDict() for model in presets.values()]
        return jsonify(models)
    

    @route("/api/loras", methods=["GET"])
    def loras(self):
        if("model" in request.args):
            loranames = self.pipelines.getLORAList(request.args["model"])
            return jsonify(loranames)


    @route("/api/async", methods=["GET"])
    def getJobAsync(self):
        # print(self.job.status)
        return jsonify(self.job.status)
    

    @route("/api/async/<action>", methods=["POST"])
    def asyncAction(self, action):
        if (self.job.thread is not None and self.job.thread.is_alive()):
            return self.getJobAsync()
        clear_output()
        r = request
        runfunc = getattr(self, f'{action}Run')
        self.job.thread = Thread(target = runfunc, args=[r.data])
        self.job.thread.start()
        self.job.status = {"status":"running", "action": action}
        # print(self.job.status)
        return jsonify(self.job.status)


    @route("/api/<action>", methods=["POST"])
    def syncAction(self, action):
        r = request
        runfunc = getattr(self, f'{action}Run')
        output = runfunc(r.data)
        return jsonify(output)


    def updateProgress(self, description, total, done):
        self.job.status['description'] = description
        self.job.status['total'] = total
        self.job.status['done'] = done


    def generateRun(self, data:bytes):
        params = GenerationParameters.from_json(data)
        # print(params)
        try:
            print('=== generate ===')
            self.prescaleBefore(params)

            outputimages = []
            for i in range(0, params.batch):
                self.updateProgress(f"Running", params.batch, i)
                outimage, usedseed = self.pipelines.generate(params)
                display(outimage)
                outimage = self.prescaleAfter([outimage], params)[0]
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"generate", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"generate", "error":str(e) }
            raise e
        

    def inpaintRun(self, data:bytes):
        params = GenerationParameters.from_json(data)
        # print(params)
        try:
            print('=== inpaint ===')
            initimageparams = params.getInitImage()
            if (initimageparams is None):
                raise Exception("No init image provided")
            if (params.getMaskImage() is None):
                maskimage = alphaToMask(initimageparams.image)
                params.setMaskImage(maskimage)

            self.prescaleBefore(params)

            outputimages = []
            for i in range(0, params.batch):
                self.updateProgress(f"Running", params.batch, i)

                outimage, usedseed = self.pipelines.generate(params)

                display(outimage)
                outimage = self.prescaleAfter([outimage], params)[0]
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"generate", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"generate", "error":str(e) }
            raise e


    def generateTiledRun(self, data:bytes):
        params = TiledGenerationParameters.from_json(data)
        # print(params)
        try:
            print('=== generateTiled ===')
            outputimages = []
            for i in range(0, params.batch):
                self.updateProgress(f"Running", params.batch, i)
                if (params.tilemethod=="singlepass"):
                    outimage, usedseed = tiledProcessorCentred(tileprocessor=tiledImageToImage, pipelines=self.pipelines, params=params, tilewidth=params.tilewidth, tileheight=params.tileheight, 
                                                               overlap=params.tileoverlap, alignmentx=params.tilealignmentx, alignmenty=params.tilealignmenty)
                elif (params.tilemethod=="multipass"):
                    outimage, usedseed = tiledImageToImageMultipass(tileprocessor=tiledImageToImage, pipelines=self.pipelines, params=params, tilewidth=params.tilewidth, tileheight=params.tileheight, 
                                                                    overlap=params.tileoverlap, passes=2, strengthMult=0.5)
                elif (params.tilemethod=="inpaint"):
                    outimage, usedseed = tiledProcessorCentred(tileprocessor=tiledInpaint, pipelines=self.pipelines, params=params, tilewidth=params.tilewidth, tileheight=params.tileheight, 
                                                               overlap=params.tileoverlap)
                else:
                    raise Exception(f"Unknown method: {params.tilemethod}")
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"img2imgTiled", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"img2imgTiled", "error":str(e) }
            raise e


    def upscaleRun(self, data:bytes):
        params = UpscaleGenerationParameters.from_json(data)
        params.generationtype = "upscale"
        # print(params)
        try:
            print('=== upscale ===')
            initimageparams = params.getInitImage()
            if (initimageparams is None):
                raise Exception("No init image provided")

            outputimages = []
            for i in range(0, params.batch):
                self.updateProgress(f"Running", params.batch, i)
                if(params.upscalemethod == "diffusers"):
                    outimage, seed = self.pipelines.generate(params)
                elif(params.upscalemethod == "esrgan"):
                    outimage = self.tools.upscaleEsrgan(initimageparams.image, scale=params.upscaleamount, model=params.models[0].name)
                else:
                    outimage = self.tools.upscaleEsrgan(initimageparams.image, scale=params.upscaleamount)
                outputimages.append({ "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"upscale", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"upscale", "error":str(e) }
            raise e
        

    def preprocessRun(self, data:bytes):
        params = GenerationParameters.from_json(data)
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

            self.job.status = { "status":"finished", "action":"preprocess", "images": [{"image":base64EncodeImage(outimage)}] }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"upscale", "error":str(e) }
            raise e


    def prescaleBefore(self, params:GenerationParameters):
        if (float(params.prescale) > 1):
            for controlimage in params.controlimages:
                controlimage.image = self.tools.upscaleEsrgan(controlimage.image, int(params.prescale), "remacri")
        elif (float(params.prescale) < 1):
            for controlimage in params.controlimages:
                controlimage.image = controlimage.image.resize((int(controlimage.image.width * float(params.prescale)), int(controlimage.image.height * float(params.prescale))), Image.LANCZOS)
        

    def prescaleAfter(self, images:List[Image.Image], params:GenerationParameters) -> List[Image.Image]:
        if (params.prescale > 1):
            prescaledimages = []
            for image in images:
                image = image.resize((int(image.width / params.prescale), int(image.height / params.prescale)), Image.LANCZOS)
                prescaledimages.append(image)
            return prescaledimages
        elif (params.prescale < 1):
            prescaledimages = []
            for image in images:
                image = self.tools.upscaleEsrgan(image, int(1 / params.prescale), "4x_remacri")
                prescaledimages.append(image)
            return prescaledimages
        else:
            return images