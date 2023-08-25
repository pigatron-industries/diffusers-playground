from flask import request, jsonify
from flask_classful import FlaskView, route
from threading import Thread
from typing import List
from .inference.DiffusersPipelines import *
from .inference.DiffusersUtils import tiledProcessorCentred, tiledImageToImageMultipass, tiledImageToImage, tiledInpaint, compositedInpaint
from .inference.arch.GenerationParameters import GenerationParameters, ModelParameters, ControlImageParameters, IMAGETYPE_INITIMAGE, IMAGETYPE_MASKIMAGE, IMAGETYPE_CONTROLIMAGE
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


    @route("/api/async", methods=["GET"])
    def getJobAsync(self):
        print(self.job.status)
        return jsonify(self.job.status)


    @route("/api/async/<action>", methods=["POST"])
    def asyncAction(self, action):
        if (self.job.thread is not None and self.job.thread.is_alive()):
            return self.getJobAsync()
        clear_output()
        r = request
        params = json.loads(r.data)
        runfunc = getattr(self, f'{action}Run')
        self.job.thread = Thread(target = runfunc, kwargs=params)
        self.job.thread.start()
        self.job.status = {"status":"running", "action": action}
        print(self.job.status)
        return jsonify(self.job.status)


    @route("/api/<action>", methods=["POST"])
    def syncAction(self, action):
        r = request
        params = json.loads(r.data)
        runfunc = getattr(self, f'{action}Run')
        output = runfunc(**params)
        return jsonify(output)


    def updateProgress(self, description, total, done):
        self.job.status['description'] = description
        self.job.status['total'] = total
        self.job.status['done'] = done

    
    def txt2imgRun(self, seed=None, prompt="", negprompt="", steps=20, scale=9, width=512, height=512, scheduler="DPMSolverMultistepScheduler", model:str="", 
                   controlimages=[], controlmodels=[], controlscales:List[float]=[1.0], batch=1, **kwargs):
        try:
            print('=== txt2img ===')
            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Width: {width}, Height: {height}, Scheduler: {scheduler}')
            print(f'Control Models: {controlmodels}')

            controlimages = base64DecodeImages(controlimages)

            # create params TODO pass in object as is to api
            controlimageparams = []
            for i in range(0, len(controlimages)):
                controlimageparams.append(ControlImageParameters(image=controlimages[i], type=IMAGETYPE_CONTROLIMAGE, model=controlmodels[i], condscale=controlscales[i]))
            params = GenerationParameters(prompt=prompt, negprompt=negprompt, steps=steps, cfgscale=scale, width=width, height=height, scheduler=scheduler, seed=seed, 
                                          models=[ModelParameters(name=model)], controlimages=controlimageparams)

            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                outimage, usedseed = self.pipelines.generate(params)
                display(outimage)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"txt2img", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"txt2img", "error":str(e) }
            raise e


    def img2imgRun(self, seed:int|None=None, prompt:str="", negprompt:str="", strength:float=0.5, scale:float=9, 
                   prescale:float=1, scheduler:str="EulerDiscreteScheduler", model:str="",
                   controlimages:List[Image.Image]=[], controlmodels:List[str]=[], controlscales:List[float]=[1.0], batch:int=1, **kwargs):
        try:
            print('=== img2img ===')
            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')
            print(f'Control Models: {controlmodels}')
            print(f'Control Images: {len(controlimages)}')

            controlimages = base64DecodeImages(controlimages)
            controlimages = self.prescaleBefore(controlimages, prescale)

            initimage = controlimages[0]
            controlimages.pop(0)

            # create params TODO pass in object as is to api
            controlimageparams = [ControlImageParameters(image=initimage, model=IMAGETYPE_INITIMAGE)]
            for i in range(0, len(controlimages)):
                controlimageparams.append(ControlImageParameters(image=controlimages[i], type=IMAGETYPE_CONTROLIMAGE, model=controlmodels[i], condscale=controlscales[i]))
            params = GenerationParameters(prompt=prompt, negprompt=negprompt, cfgscale=scale, strength=strength, width=initimage.width, height=initimage.height, scheduler=scheduler, seed=seed, 
                                          models=[ModelParameters(name=model)], controlimages=controlimageparams)

            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                outimage, usedseed = self.pipelines.generate(params)
                display(outimage)
                outimage = self.prescaleAfter([outimage], prescale)[0]
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"img2img", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"img2img", "error":str(e) }
            raise e


    def img2imgTiledRun(self, controlimages, seed=None, prompt="", negprompt="", strength=0.4, scale=9, scheduler="EulerDiscreteScheduler", model=None, controlmodels=None,
                        method="singlepass", tilealignmentx="tile_centre", tilealignmenty="tile_centre", tilewidth=640, tileheight=640, tileoverlap=128, batch=1, **kwargs):
        try:
            print('=== img2imgTiled ===')
            print(f'Method: {method}')
            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')
            print(f'Method: {method}, Tile width: {tilewidth}, Tile height: {tileheight}, Tile Overlap: {tileoverlap}')

            controlimages = base64DecodeImages(controlimages)
            initimage = controlimages[0]
            controlimages.pop(0)

            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                if (method=="singlepass"):
                    outimage, usedseed = tiledProcessorCentred(tileprocessor=tiledImageToImage, pipelines=self.pipelines, initimage=initimage, controlimages=controlimages, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                  scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap, 
                                                                  alignmentx=tilealignmentx, alignmenty=tilealignmenty, model=model, controlmodels=controlmodels, callback=self.updateProgress)
                elif (method=="multipass"):
                    outimage, usedseed = tiledImageToImageMultipass(tileprocessor=tiledImageToImage, pipelines=self.pipelines, initimage=initimage, controlimages=controlimages, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                    scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap, 
                                                                    passes=2, strengthMult=0.5, model=model, controlmodels=controlmodels, callback=self.updateProgress)
                elif (method=="inpaint"):
                    outimage, usedseed = tiledProcessorCentred(tileprocessor=tiledInpaint, pipelines=self.pipelines, initimage=initimage, controlimages=controlimages, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                       scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, 
                                                                       model=model, controlmodels=controlmodels, overlap=tileoverlap)
                else:
                    raise Exception(f"Unknown method: {method}")
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"img2imgTiled", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"img2imgTiled", "error":str(e) }
            raise e


    def inpaintRun(self, controlimages, maskimage=None, seed=None, prompt="", negprompt="", steps=30, scale=9, prescale:float=1, strength=1.0, 
                   scheduler="EulerDiscreteScheduler", model:str="", controlmodels=None, controlscales:List[float]=[1.0], batch=1, **kwargs):
        try:
            print('=== inpaint ===')
            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Scheduler: {scheduler}')

            controlimages = base64DecodeImages(controlimages)
            initimage = controlimages[0]
            if maskimage is None:
                maskimage = alphaToMask(initimage)
            else:
                maskimage = base64DecodeImage(maskimage)
            if(controlmodels is not None and len(controlmodels) > 0):
                controlimages.pop(0)
            else:
                controlimages = []
                controlmodels = []

            maskimage = self.prescaleBefore([maskimage], prescale)[0]
            initimage = self.prescaleBefore([initimage], prescale)[0]
            controlimages = self.prescaleBefore(controlimages, prescale)

            controlimageparams = [ControlImageParameters(image=initimage, model=IMAGETYPE_INITIMAGE), ControlImageParameters(image=maskimage, model=IMAGETYPE_MASKIMAGE)]
            for i in range(0, len(controlimages)):
                controlimageparams.append(ControlImageParameters(image=controlimages[i], type=IMAGETYPE_CONTROLIMAGE, model=controlmodels[i], condscale=controlscales[i]))
            params = GenerationParameters(prompt=prompt, negprompt=negprompt, cfgscale=scale, width=initimage.width, height=initimage.height, scheduler=scheduler, seed=seed, 
                                          models=[ModelParameters(name=model)], controlimages=controlimageparams)

            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                outimage, usedseed = compositedInpaint(self.pipelines, params)
                # outimage = applyColourCorrection(initimage, outimage)
                outimage = self.prescaleAfter([outimage], prescale)[0]
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"inpaint", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"inpaint", "error":str(e) }
            raise e


    def upscaleRun(self, controlimages, method="esrgan/remacri", amount=4, prompt="", negprompt="", steps=30, scale=7.0, scheduler="EulerDiscreteScheduler", batch=1, model="stabilityai/stable-diffusion-x4-upscaler", **kwargs):
        try:
            print('=== upscale ===')
            print(f'Method: {method}')
            if(model is None):
                model = "stabilityai/stable-diffusion-x4-upscaler"
            if(steps == 0):
                steps = 75

            controlimages = base64DecodeImages(controlimages)
            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                if(method == "stable-diffusion"):
                    outimage, seed = self.pipelines.upscale(initimage=controlimages[0], prompt=prompt, negprompt=negprompt, scheduler=scheduler, scale=scale, steps=steps, model=model)
                elif(method.startswith("esrgan")):
                    outimage = self.tools.upscaleEsrgan(controlimages[0], scale=amount, model=method.split('/')[1])
                else:
                    outimage = self.tools.upscaleEsrgan(controlimages[0], scale=amount)
                outputimages.append({ "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"upscale", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"upscale", "error":str(e) }
            raise e
        

    def preprocessRun(self, controlimages, process, **kwargs):
        try:
            print('=== preprocess ===')
            print(f'Process: {process}')

            controlimages = base64DecodeImages(controlimages)
            processor = str_to_class(process + 'Processor')()
            
            pipeline = ProcessingPipelineBuilder.fromImage(controlimages[0])
            pipeline.addTask(processor)
            outimage = pipeline()

            self.job.status = { "status":"finished", "action":"preprocess", "images": [{"image":base64EncodeImage(outimage)}] }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"upscale", "error":str(e) }
            raise e


    def prescaleBefore(self, images:List[Image.Image], prescale:float) -> List[Image.Image]:
        if (float(prescale) > 1):
            prescaledimages = []
            for image in images:
                image = self.tools.upscaleEsrgan(image, int(prescale), "remacri")
                prescaledimages.append(image)
            return prescaledimages
        elif (float(prescale) < 1):
            prescaledimages = []
            for image in images:
                image = image.resize((int(image.width * float(prescale)), int(image.height * float(prescale))), Image.LANCZOS)
                prescaledimages.append(image)
            return prescaledimages
        else:
            return images
        

    def prescaleAfter(self, images:List[Image.Image], prescale:float) -> List[Image.Image]:
        if (float(prescale) > 1):
            prescaledimages = []
            for image in images:
                image = image.resize((int(image.width / float(prescale)), int(image.height / float(prescale))), Image.LANCZOS)
                prescaledimages.append(image)
            return prescaledimages
        elif (float(prescale) < 1):
            prescaledimages = []
            for image in images:
                image = self.tools.upscaleEsrgan(image, int(1 / float(prescale)), "remacri")
                prescaledimages.append(image)
            return prescaledimages
        else:
            return images