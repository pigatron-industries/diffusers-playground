from flask import request, jsonify
from flask_classful import FlaskView, route
from threading import Thread
from .inference.DiffusersPipelines import *
from .inference.DiffusersUtils import tiledImageToImageCentred, tiledImageToImageMultipass, tiledInpaint, tiledImageToImageInpaintSeams, compositedInpaint
from .ImageUtils import base64EncodeImage, base64DecodeImage, alphaToMask, applyColourCorrection
from .ImageTools import ImageTools
import json

from IPython.display import display


class DiffusersJob():
    def __init__(self):
        self.thread = None
        self.status = { "status":"none", "action":"none" }


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
        models = [model.toDict() for model in self.pipelines.presets.models.values()]
        return jsonify(models)


    @route("/api/async", methods=["GET"])
    def getJobAsync(self):
        print(self.job.status)
        return jsonify(self.job.status)


    @route("/api/async/<action>", methods=["POST"])
    def asyncAction(self, action):
        if (self.job.thread is not None and self.job.thread.is_alive()):
            return self.getJobAsync()
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


    def updateAsyncProgress(self, description, total, done):
        self.job.status['description'] = description
        self.job.status['total'] = total
        self.job.status['done'] = done

    
    def txt2imgRun(self, seed=None, prompt="", negprompt="", steps=20, scale=9, width=512, height=512, scheduler="DPMSolverMultistepScheduler", model=None, batch=1, **kwargs):
        try:
            print('=== txt2img ===')
            if(self.pipelines.pipelineImageToImage is None and model is None):
                model = DEFAULT_TEXTTOIMAGE_MODEL
            if(model is not None and model != ""):
                print(f'Model: {model}')
                self.updateAsyncProgress(f"Loading Model ${model}", 1, 0)
                self.pipelines.createTextToImagePipeline(model)

            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Width: {width}, Height: {height}, Scheduler: {scheduler}')

            outputimages = []
            for i in range(0, batch):
                self.updateAsyncProgress(f"Running", 1, 0)
                outimage, usedseed = self.pipelines.textToImage(prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, width=width, height=height, scheduler=scheduler, seed=seed)
                display(outimage)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"txt2img", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"txt2img", "error":e.message }
            raise e


    def img2imgRun(self, initimage, seed=None, prompt="", negprompt="", strength=0.5, scale=9, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        try:
            print('=== img2img ===')
            initimage = base64DecodeImage(initimage)

            if(self.pipelines.pipelineImageToImage is None and model is None):
                model = DEFAULT_TEXTTOIMAGE_MODEL
            if(model is not None and model != ""):
                print(f'Model: {model}')
                self.pipelines.createImageToImagePipeline(model)

            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

            outputimages = []
            for i in range(0, batch):
                outimage, usedseed = self.pipelines.imageToImage(inimage=initimage, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, seed=seed, scheduler=scheduler)
                outimage = applyColourCorrection(initimage, outimage)
                display(outimage)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"img2img", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"img2img", "error":e.message }
            raise e


    def depth2imgRun(self, initimage, seed=None, prompt="", negprompt="", strength=0.5, scale=9, steps=50, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        try:
            print('=== depth2img ===')
            initimage = base64DecodeImage(initimage)

            if(self.pipelines.pipelineDepthToImage is None and model is None):
                model = DEFAULT_DEPTHTOIMAGE_MODEL
            if(model is not None and model != ""):
                print(f'Model: {model}')
                self.pipelines.createDepthToImagePipeline(model)

            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

            outputimages = []
            for i in range(0, batch):
                outimage, usedseed = self.pipelines.depthToImage(inimage=initimage, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, steps=steps, seed=seed, scheduler=scheduler)
                display(outimage)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"depth2img", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"depth2img", "error":e.message }
            raise e


    def imagevariationRun(self, initimage, seed=None, steps=30, scale=9, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        try:
            print('=== imagevar ===')
            initimage = base64DecodeImage(initimage)

            if(self.pipelines.pipelineImageVariation is None and model is None):
                model = DEFAULT_IMAGEVARIATION_MODEL
            if(model is not None and model != ""):
                print(f'Model: {model}')
                self.pipelines.createImageVariationPipeline(model)

            print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Scheduler: {scheduler}')

            outputimages = []
            for i in range(0, batch):
                outimage, usedseed = self.pipelines.imageVariation(initimage=initimage, steps=steps, scale=scale, seed=seed, scheduler=scheduler)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"imagevariation", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"imagevariation", "error":e.message }
            raise e


    def instructpix2pixRun(self, initimage, prompt, seed=None, steps=30, scale=9, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        try:
            print('=== instructpix2pix ===')
            initimage = base64DecodeImage(initimage)

            if(self.pipelines.pipelineInstructPixToPix is None and model is None):
                model = DEFAULT_INSTRUCTPIXTOPIX_MODEL
            if(model is not None and model != ""):
                print(f'Model: {model}')
                self.pipelines.createInstructPixToPixPipeline(model)

            print(f'Prompt: {prompt}')
            print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Scheduler: {scheduler}')

            outputimages = []
            for i in range(0, batch):
                outimage, usedseed = self.pipelines.instructPixToPix(prompt=prompt, initimage=initimage, steps=steps, scale=scale, seed=seed, scheduler=scheduler)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"instructpix2pix", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"instructpix2pix", "error":e.message }
            raise e


    def img2imgTiledRun(self, initimage, seed=None, prompt="", negprompt="", strength=0.4, scale=9, scheduler="EulerDiscreteScheduler", model=None, 
                        method="singlepass", tilealignmentx="tile_centre", tilealignmenty="tile_centre", tilewidth=640, tileheight=640, tileoverlap=128, batch=1, **kwargs):
        try:
            print('=== img2imgTiled ===')
            initimage = base64DecodeImage(initimage)

            if(self.pipelines.pipelineImageToImage is None and model is None):
                model = DEFAULT_TEXTTOIMAGE_MODEL
            if(model is not None and model != ""):
                print(f'Model: {model}')
                self.pipelines.createImageToImagePipeline(model)

            print(f'Method: {method}')
            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')
            print(f'Method: {method}, Tile width: {tilewidth}, Tile height: {tileheight}, Tile Overlap: {tileoverlap}')

            outputimages = []
            for i in range(0, batch):
                if (method=="singlepass"):
                    outimage, usedseed = tiledImageToImageCentred(self.pipelines, initimg=initimage, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                  scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap, 
                                                                  alignmentx=tilealignmentx, alignmenty=tilealignmenty)
                elif (method=="multipass"):
                    outimage, usedseed = tiledImageToImageMultipass(self.pipelines, initimg=initimage, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                    scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap, 
                                                                    passes=2, strengthMult=0.5)
                elif (method=="inpaint_seams"):
                    outimage, usedseed = tiledImageToImageInpaintSeams(self.pipelines, initimg=initimage, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                    scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"img2imgTiled", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"img2imgTiled", "error":e.message }
            raise e


    def inpaintRun(self, initimage, maskimage=None, seed=None, prompt="", negprompt="", steps=30, scale=9, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        try:
            print('=== inpaint ===')
            initimage = base64DecodeImage(initimage)
            if maskimage is None:
                maskimage = alphaToMask(initimage)
            else:
                maskimage = base64DecodeImage(maskimage)

            if(self.pipelines.pipelineInpainting is None and model is None):
                model = DEFAULT_INPAINT_MODEL
            if(model is not None and model != ""):
                print(f'Model: {model}')
                self.pipelines.createInpaintPipeline(model)

            print(f'Prompt: {prompt}')
            print(f'Negative: {negprompt}')
            print(f'Height: {initimage.height}, Width: {initimage.width}')
            print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Scheduler: {scheduler}')

            outputimages = []
            for i in range(0, batch):
                if(initimage.height > 512 or initimage.width > 512):
                    outimage, usedseed = tiledInpaint(self.pipelines, initimg=initimage, maskimg=maskimage, prompt=prompt, negprompt=negprompt, steps=steps, 
                                                    scale=scale, scheduler=scheduler, seed=seed, overlap=128)
                else:
                    outimage, usedseed = compositedInpaint(self.pipelines, initimage=initimage, maskimage=maskimage, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, seed=seed, scheduler=scheduler)
                    outimage = applyColourCorrection(initimage, outimage)
                outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"inpaint", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"inpaint", "error":e.message }
            raise e


    def upscaleRun(self, initimage, method="esrgan/remacri", amount=4, prompt="", scheduler="EulerDiscreteScheduler", batch=1, **kwargs):
        try:
            print('=== upscale ===')
            initimage = base64DecodeImage(initimage)

            print(f'Method: {method}')

            outputimages = []
            for i in range(0, batch):
                if(method == "stable-difusion"):
                    outimage = self.pipelines.upscale(inimage=initimage, prompt=prompt, scheduler=scheduler)
                elif(method.startswith("esrgan")):
                    outimage = self.tools.upscaleEsrgan(initimage, scale=amount, model=method.split('/')[1])
                else:
                    outimage = self.tools.upscaleEsrgan(initimage, scale=amount)
                outputimages.append({ "image": base64EncodeImage(outimage) })

            self.job.status = { "status":"finished", "action":"upscale", "images": outputimages }
            return self.job.status

        except Exception as e:
            self.job.status = { "status":"error", "action":"upscale", "error":e.message }
            raise e