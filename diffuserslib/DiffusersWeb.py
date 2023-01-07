from flask import request, jsonify
from flask_classful import FlaskView, route
from .DiffusersPipelines import DiffusersPipelines
from .DiffusersUtils import tiledImageToImageOffset, tiledImageToImageMultipass, tiledInpaint, tiledImageToImageInpaintSeams
from .ImageUtils import base64EncodeImage, base64DecodeImage, alphaToMask, compositeImages
from .ImageTools import ImageTools
import json

from IPython.display import display


class DiffusersView(FlaskView):
    route_base = '/'
    pipelines: DiffusersPipelines = None
    tools: ImageTools = None

    def __init__(self):
        pass


    @route("/api/", methods=["GET"])
    def info(self):
        return 'stable-diffusion'


    @route("/api/txt2img", methods=["POST"])
    def txt2img(self):
        r = request
        data = json.loads(r.data)
        seed = data.get("seed", None)
        prompt = data.get("prompt", "")
        negprompt = data.get("negprompt", "")
        steps = data.get("steps", 20)
        scale = data.get("scale", 9)
        width = data.get("width", 512)
        height = data.get("height", 512)
        scheduler = data.get("scheduler", "DPMSolverMultistepScheduler")
        batch = data.get("batch", 1)

        print('txt2img')
        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Width: {width}, Height: {height}, Scheduler: {scheduler}')

        outputimages = []
        for i in range(0, batch):
            outimage, usedseed = self.pipelines.textToImage(prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, width=width, height=height, scheduler=scheduler, seed=seed)
            display(outimage)
            outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        output = { "images": outputimages }
        return jsonify(output)

    
    @route("/api/img2img", methods=["POST"])
    def img2img(self):
        r = request
        data = json.loads(r.data)
        seed = data.get("seed", None)
        prompt = data.get("prompt", "")
        negprompt = data.get("negprompt", "")
        strength = data.get("strength", 0.5)
        scale = data.get("scale", 9)
        scheduler = data.get("scheduler", "EulerDiscreteScheduler")
        batch = data.get("batch", 1)
        initimage = base64DecodeImage(data['initimage'])

        print('img2img')
        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

        outputimages = []
        for i in range(0, batch):
            outimage, usedseed = self.pipelines.imageToImage(inimage=initimage, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, seed=seed, scheduler=scheduler)
            display(outimage)
            outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        output = { "images": outputimages }
        return jsonify(output)


    @route("/api/depth2img", methods=["POST"])
    def depth2img(self):
        r = request
        data = json.loads(r.data)
        seed = data.get("seed", None)
        prompt = data.get("prompt", "")
        negprompt = data.get("negprompt", "")
        strength = data.get("strength", 0.5)
        scale = data.get("scale", 9)
        scheduler = data.get("scheduler", "EulerDiscreteScheduler")
        batch = data.get("batch", 1)
        initimage = base64DecodeImage(data['initimage'])

        print('depth2img')
        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

        outputimages = []
        for i in range(0, batch):
            outimage, usedseed = self.pipelines.depthToImage(inimage=initimage, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, seed=seed, scheduler=scheduler)
            display(outimage)
            outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        output = { "images": outputimages }
        return jsonify(output)


    @route("/api/img2imgTiled", methods=["POST"])
    def img2imgTiled(self):
        r = request
        data = json.loads(r.data)
        seed = data.get("seed", None)
        prompt = data.get("prompt", "")
        negprompt = data.get("negprompt", "")
        strength = data.get("strength", 0.4)
        steps = data.get("steps", 50)
        scale = data.get("scale", 9)
        scheduler = data.get("scheduler", "EulerDiscreteScheduler")
        method = data.get("method", "multipass")
        offsetx = data.get("offsetx", 0)
        offsety = data.get("offsety", 0)
        batch = data.get("batch", 1)
        initimage = base64DecodeImage(data['initimage'])

        print('img2imgTiled')
        print(f'Method: {method}')
        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

        outputimages = []
        for i in range(0, batch):
            if (method=="singlepass"):
                outimage, usedseed = tiledImageToImageOffset(self.pipelines, initimg=initimage, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                             scale=scale, scheduler=scheduler, seed=seed, tilewidth=640, tileheight=640, overlap=128, 
                                                             offsetx=offsetx, offsety=offsety)
            elif (method=="multipass"):
                outimage, usedseed = tiledImageToImageMultipass(self.pipelines, initimg=initimage, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                scale=scale, scheduler=scheduler, seed=seed, tilewidth=640, tileheight=640, overlap=128, 
                                                                passes=2, strengthMult=0.5)
            elif (method=="inpaint_seams"):
                outimage, usedseed = tiledImageToImageInpaintSeams(self.pipelines, initimg=initimage, prompt=prompt, negprompt=negprompt, strength=strength, 
                                                                   scale=scale, scheduler=scheduler, seed=seed, tilewidth=640, tileheight=640, overlap=128)
            outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        output = { "images": outputimages }
        return jsonify(output)


    @route("/api/inpaint", methods=["POST"])
    def inpaint(self):
        r = request
        data = json.loads(r.data)
        seed = data.get("seed", None)
        prompt = data.get("prompt", "")
        negprompt = data.get("negprompt", "")
        steps = data.get("steps", 30)
        scale = data.get("scale", 9)
        scheduler = data.get("scheduler", "EulerDiscreteScheduler")
        batch = data.get("batch", 1)
        initimage = base64DecodeImage(data['initimage'])
        if "maskimage" in data:
            maskimage = base64DecodeImage(data['maskimage'])
        else:
            maskimage = alphaToMask(initimage);

        print('inpaint')
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
                outimage, usedseed = self.pipelines.inpaint(inimage=initimage, maskimage=maskimage, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, seed=seed, scheduler=scheduler)
                outimage = compositeImages(outimage, initimage, maskimage)
            outputimages.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        output = { "images": outputimages }
        return jsonify(output)


    @route("/api/upscale", methods=["POST"])
    def upscale(self):
        r = request
        data = json.loads(r.data)
        method = data.get("method", "esrgan/remacri")
        amount = data.get("amount", 4)
        prompt = data.get("prompt", "")
        scheduler = data.get("scheduler", "EulerDiscreteScheduler")
        batch = data.get("batch", 1)
        initimage = base64DecodeImage(data['initimage'])

        print('upscale')
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

        output = { "images": outputimages }
        return jsonify(output)
