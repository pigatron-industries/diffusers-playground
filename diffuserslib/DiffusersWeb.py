from flask import request, jsonify
from flask_classful import FlaskView, route
from threading import Thread
from .inference.DiffusersPipelines import *
from .inference.DiffusersUtils import tiledImageToImageCentred, tiledImageToImageMultipass, tiledImageToImageInpaintSeams, compositedInpaint
from .ImageUtils import base64EncodeImage, base64DecodeImage, base64DecodeImages, alphaToMask, applyColourCorrection
from .imagetools.ImageTools import ImageTools
from .processing.TransformerProcessors import *
from .processing.FilterProcessors import *
from .processing.ProcessingPipelineFactory import ProcessingPipelineBuilder
import json

from IPython.display import display, clear_output

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class DiffusersJob():
    def __init__(self):
        self.thread = None
        self.status = { "status":"none", "action":"none", "description": "", "total": 0, "done": 0}


class DiffusersView(FlaskView):
    route_base = '/'
    pipelines: DiffusersPipelines = None
    tools: ImageTools = None
    job: DiffusersJob = DiffusersJob()

    def __init__(self):
        pass


    @route("/api/", methods=["GET"])
    def info(self):
        return 'stable-diffusion'


    @route("/api/models", methods=["GET"])
    def models(self):
        presets = self.pipelines.presetsImage
        if("type" in request.args):
          if(request.args["type"] == "inpaint"):
              presets = self.pipelines.presetsInpaint
          elif(request.args["type"] == "control"):
              presets = self.pipelines.presetsControl
        models = [model.toDict() for model in presets.models.values()]
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

    
    def txt2imgRun(self, seed=None, prompt="", negprompt="", steps=20, scale=9, width=512, height=512, scheduler="DPMSolverMultistepScheduler", model=None, controlimages=None, controlmodels=None, batch=1, **kwargs):
        try:
            print('=== txt2img ===')
            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Width: {width}, Height: {height}, Scheduler: {scheduler}')
            print(f'Control Models: {controlmodels}')

            controlimages = base64DecodeImages(controlimages)
            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                if(controlimages is None or len(controlimages) == 0):
                    outimage, usedseed = self.pipelines.textToImage(prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, width=width, height=height, scheduler=scheduler, seed=seed, model=model)
                else:
                    outimage, usedseed = self.pipelines.textToImageControlNet(prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, width=width, height=height, scheduler=scheduler, seed=seed, model=model, controlimage=controlimages, controlmodel=controlmodels)
                display(outimage)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"txt2img", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"txt2img", "error":str(e) }
            raise e


    def img2imgRun(self, seed=None, prompt="", negprompt="", strength=0.5, scale=9, scheduler="EulerDiscreteScheduler", model=None, controlimages=None, controlmodels=None, batch=1, **kwargs):
        try:
            print('=== img2img ===')
            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')
            print(f'Control Models: {controlmodels}')
            print(f'Control Images: {len(controlimages)}')

            controlimages = base64DecodeImages(controlimages)
            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                if(len(controlimages) == 1):
                    outimage, usedseed = self.pipelines.imageToImage(initimage=controlimages[0], prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, seed=seed, scheduler=scheduler, model=model)
                else:
                    initimage=controlimages[0]
                    controlimages.pop(0)
                    outimage, usedseed = self.pipelines.imageToImageControlNet(initimage=initimage, controlimage=controlimages, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, seed=seed, scheduler=scheduler, model=model, controlmodel=controlmodels)
                display(outimage)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"img2img", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"img2img", "error":str(e) }
            raise e


    def depth2imgRun(self, controlimages, seed=None, prompt="", negprompt="", strength=0.5, scale=9, steps=50, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        try:
            print('=== depth2img ===')
            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

            controlimages = base64DecodeImages(controlimages)
            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                outimage, usedseed = self.pipelines.depthToImage(inimage=controlimages[0], prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, steps=steps, seed=seed, scheduler=scheduler, model=model)
                display(outimage)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"depth2img", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"depth2img", "error":str(e) }
            raise e


    def imagevariationRun(self, controlimages, seed=None, steps=30, scale=9, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        try:
            print('=== imagevar ===')
            print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Scheduler: {scheduler}')

            controlimages = base64DecodeImages(controlimages)
            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                outimage, usedseed = self.pipelines.imageVariation(initimage=controlimages[0], steps=steps, scale=scale, seed=seed, scheduler=scheduler, model=model)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"imagevariation", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"imagevariation", "error":str(e) }
            raise e


    def instructpix2pixRun(self, controlimages, prompt, seed=None, steps=30, scale=9, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        try:
            print('=== instructpix2pix ===')
            print(f'Prompt: {prompt}')
            print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Scheduler: {scheduler}')

            controlimages = base64DecodeImages(controlimages)
            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                outimage, usedseed = self.pipelines.instructPixToPix(prompt=prompt, initimage=controlimages[0], steps=steps, scale=scale, seed=seed, scheduler=scheduler, model=model)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"instructpix2pix", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"instructpix2pix", "error":str(e) }
            raise e


    def img2imgTiledRun(self, controlimages, seed=None, prompt="", negprompt="", strength=0.4, scale=9, scheduler="EulerDiscreteScheduler", model=None, 
                        method="singlepass", tilealignmentx="tile_centre", tilealignmenty="tile_centre", tilewidth=640, tileheight=640, tileoverlap=128, batch=1, **kwargs):
        try:
            print('=== img2imgTiled ===')
            print(f'Method: {method}')
            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')
            print(f'Method: {method}, Tile width: {tilewidth}, Tile height: {tileheight}, Tile Overlap: {tileoverlap}')

            controlimages = base64DecodeImage(controlimages)
            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                if (method=="singlepass"):
                    outimage, usedseed = tiledImageToImageCentred(self.pipelines, initimg=controlimages[0], prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                  scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap, 
                                                                  alignmentx=tilealignmentx, alignmenty=tilealignmenty, model=model, callback=self.updateProgress)
                elif (method=="multipass"):
                    outimage, usedseed = tiledImageToImageMultipass(self.pipelines, initimg=controlimages[0], prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                    scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap, 
                                                                    passes=2, strengthMult=0.5, model=model, callback=self.updateProgress)
                elif (method=="inpaint_seams"):
                    outimage, usedseed = tiledImageToImageInpaintSeams(self.pipelines, initimg=controlimages[0], prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                       scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, 
                                                                       model=model, overlap=tileoverlap)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"img2imgTiled", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"img2imgTiled", "error":str(e) }
            raise e


    def inpaintRun(self, controlimages, maskimage=None, seed=None, prompt="", negprompt="", steps=30, scale=9, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        try:
            print('=== inpaint ===')
            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Scheduler: {scheduler}')

            controlimage = base64DecodeImage(controlimages)[0]
            if maskimage is None:
                maskimage = alphaToMask(controlimage)
            else:
                maskimage = base64DecodeImage(maskimage)

            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                outimage, usedseed = compositedInpaint(self.pipelines, initimage=controlimage, maskimage=maskimage, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, seed=seed, scheduler=scheduler, model=model)
                # outimage = applyColourCorrection(initimage, outimage)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"inpaint", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"inpaint", "error":str(e) }
            raise e


    def upscaleRun(self, controlimages, method="esrgan/remacri", amount=4, prompt="", scheduler="EulerDiscreteScheduler", batch=1, model=None, **kwargs):
        try:
            print('=== upscale ===')
            print(f'Method: {method}')

            controlimages = base64DecodeImages(controlimages)
            outputimages = []
            for i in range(0, batch):
                self.updateProgress(f"Running", batch, i)
                if(method == "stable-difusion"):
                    outimage = self.pipelines.upscale(inimage=controlimages[0], prompt=prompt, scheduler=scheduler, model=model)
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
