from flask import request, jsonify
from flask_classful import FlaskView, route
from threading import Thread
from .DiffusersPipelines import DiffusersPipelines
from .DiffusersUtils import tiledImageToImageOffset, tiledImageToImageMultipass, tiledInpaint, tiledImageToImageInpaintSeams, compositedInpaint
from .ImageUtils import base64EncodeImage, base64DecodeImage, alphaToMask, compositeImages
from .ImageTools import ImageTools
import json

from IPython.display import display


class DiffusersView(FlaskView):
    route_base = '/'
    pipelines: DiffusersPipelines = None
    tools: ImageTools = None

    def __init__(self):
        self.jobThread = None
        self.jobStatus = { "status":"none", "action":"none" }
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
        return jsonify(self.jobStatus)


    @route("/api/async/<action>", methods=["POST"])
    def asyncAction(self, action):
        # TODO error if job is already running
        r = request
        params = json.loads(r.data)
        runfunc = getattr(self, f'{action}Run')
        self.jobThread = Thread(target = runfunc, args=params)
        self.jobThread.start()
        self.jobStatus = {"status":"running", "action": "txt2img"}
        return jsonify(self.jobStatus)


    @route("/api/<action>", methods=["POST"])
    def syncAction(self, action):
        r = request
        params = json.loads(r.data)
        runfunc = getattr(self, f'{action}Run')
        output = runfunc(**params)
        return jsonify(output)

    
    def txt2imgRun(self, seed=None, prompt="", negprompt="", steps=20, scale=9, width=512, height=512, scheduler="DPMSolverMultistepScheduler", model=None, batch=1, **kwargs):
        print('=== txt2img ===')
        if(model is not None and model != ""):
            print(f'Model: {model}')
            self.pipelines.createTextToImagePipeline(model)

        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Width: {width}, Height: {height}, Scheduler: {scheduler}')

        outputimages = []
        for i in range(0, batch):
            outimage, usedseed = self.pipelines.textToImage(prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, width=width, height=height, scheduler=scheduler, seed=seed)
            display(outimage)
            outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        self.jobStatus = { "status":"finished", "action":"txt2img", "images": outputimages }
        return self.jobStatus


    def img2imgRun(self, initimage, seed=None, prompt="", negprompt="", strength=0.5, scale=9, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        print('=== img2img ===')
        initimage = base64DecodeImage(initimage)

        if(model is not None and model != ""):
            print(f'Model: {model}')
            self.pipelines.createImageToImagePipeline(model)

        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

        outputimages = []
        for i in range(0, batch):
            outimage, usedseed = self.pipelines.imageToImage(inimage=initimage, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, seed=seed, scheduler=scheduler)
            display(outimage)
            outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        self.jobStatus = { "status":"finished", "action":"img2img", "images": outputimages }
        return self.jobStatus


    def depth2imgRun(self, initimage, seed=None, prompt="", negprompt="", strength=0.5, scale=9, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        print('=== depth2img ===')
        initimage = base64DecodeImage(initimage)

        if(model is not None and model != ""):
            print(f'Model: {model}')
            self.pipelines.createDepthToImagePipeline(model)

        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

        outputimages = []
        for i in range(0, batch):
            outimage, usedseed = self.pipelines.depthToImage(inimage=initimage, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, seed=seed, scheduler=scheduler)
            display(outimage)
            outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        self.jobStatus = { "status":"finished", "action":"depth2img", "images": outputimages }
        return self.jobStatus


    def img2imgTiledRun(self, initimage, seed=None, prompt="", negppompt="", strength=0.4, scale=9, scheduler="EulerDiscreteScheduler", model=None, 
                        method="singlepass", offsetx=0, offsety=0, tilewidth=640, tileheight=640, tileoverlap=128, batch=1, **kwargs):
        print('=== img2imgTiled ===')
        initimage = base64DecodeImage(initimage)

        if(model is not None and model != ""):
            print(f'Model: {model}')
            self.pipelines.createImageToImagePipeline(model)

        print(f'Method: {method}')
        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

        outputimages = []
        for i in range(0, batch):
            if (method=="singlepass"):
                outimage, usedseed = tiledImageToImageOffset(self.pipelines, initimg=initimage, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                             scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap, 
                                                             offsetx=offsetx, offsety=offsety)
            elif (method=="multipass"):
                outimage, usedseed = tiledImageToImageMultipass(self.pipelines, initimg=initimage, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap, 
                                                                passes=2, strengthMult=0.5)
            elif (method=="inpaint_seams"):
                outimage, usedseed = tiledImageToImageInpaintSeams(self.pipelines, initimg=initimage, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                   scale=scale, scheduler=scheduler, seed=seed, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap)
            outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        self.jobStatus = { "status":"finished", "action":"img2imgTiled", "images": outputimages }
        return self.jobStatus


    def inpaintRun(self, initimage, maskimage=None, seed=None, prompt="", negprompt="", steps=30, scale=9, scheduler="EulerDiscreteScheduler", model=None, batch=1, **kwargs):
        print('=== inpaint ===')
        initimage = base64DecodeImage(initimage)
        if maskimage is None:
            maskimage = alphaToMask(initimage)
        else:
            maskimage = base64DecodeImage(maskimage)

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
            outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        self.jobStatus = { "status":"finished", "action":"inpaint", "images": outputimages }
        return self.jobStatus


    def upscaleRun(self, initimage, method="esrgan/remacri", amount=4, prompt="", scheduler="EulerDiscreteScheduler", batch=1, **kwargs):
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

        self.jobStatus = { "status":"finished", "action":"upscale", "images": outputimages }
        return self.jobStatus
