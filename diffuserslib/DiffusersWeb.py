from flask import Flask, Response, request, send_file, jsonify
from flask_classful import FlaskView, route
from .DiffusersPipelines import DiffusersPipelines
from .DiffusersUtils import tiledImageToImage
from .ImageUtils import base64EncodeImage, base64DecodeImage
import json

from IPython.display import display


app = Flask(__name__)


class DiffusersView(FlaskView):
    route_base = '/'
    pipelines = None

    def __init__(self):
        pass


    def run(self):
        app.run()


    @route("/", methods=["GET"])
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

        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Width: {width}, Height: {height}, Scheduler: {scheduler}')

        output = []
        for i in range(0, batch):
            outimage, usedseed = self.pipelines.textToImage(prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, width=width, height=height, scheduler=scheduler, seed=seed)
            display(outimage)
            output.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

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

        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

        output = []
        for i in range(0, batch):
            outimage, usedseed = self.pipelines.imageToImage(inimage=initimage, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, seed=seed, scheduler=scheduler)
            display(outimage)
            output.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        return jsonify(output)


    @route("/api/img2imgTiled", methods=["POST"])
    def img2imgTiled(self):
        r = request
        data = json.loads(r.data)
        seed = data.get("seed", None)
        prompt = data.get("prompt", "")
        negprompt = data.get("negprompt", "")
        strength = data.get("strength", 0.4)
        scale = data.get("scale", 9)
        scheduler = data.get("scheduler", "EulerDiscreteScheduler")
        batch = data.get("batch", 1)
        initimage = base64DecodeImage(data['initimage'])

        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Strength: {strength}, Scheduler: {scheduler}')

        output = []
        for i in range(0, batch):
            outimage = tiledImageToImage(self.pipelines, initimg=initimage, prompt=prompt, negprompt=negprompt, strength=strength, scale=scale, scheduler=scheduler, seed=seed, tilewidth=640, tileheight=640, overlap=128)
            output.append({ "image": base64EncodeImage(outimage) })

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
        maskimage = base64DecodeImage(data['maskimage'])

        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Scheduler: {scheduler}')

        output = []
        for i in range(0, batch):
            outimage, usedseed = self.pipelines.inpaint(inimage=initimage, maskimage=maskimage, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, seed=seed, scheduler=scheduler)
            display(outimage)
            output.append({ "seed": usedseed, "image": base64EncodeImage(outimage) })

        return jsonify(output)


    @route("/api/upscale", methods=["POST"])
    def upscale(self):
        r = request
        data = json.loads(r.data)
        method = data.get("method", "esrgan/remacri")
        amount = data.get("amount", 4)
        seed = data.get("seed", None)
        prompt = data.get("prompt", "")
        negprompt = data.get("negprompt", "")
        steps = data.get("steps", 30)
        scale = data.get("scale", 9)
        scheduler = data.get("scheduler", "EulerDiscreteScheduler")
        batch = data.get("batch", 1)
        initimage = base64DecodeImage(data['initimage'])
        maskimage = base64DecodeImage(data['maskimage'])

        print(f'Prompt: {prompt}')
        print(f'Negative: {negprompt}')
        print(f'Seed: {seed}, Scale: {scale}, Steps: {steps}, Scheduler: {scheduler}')

        output = []
        for i in range(0, batch):
            pass
            # TODO
            # outimage = upscale(inimage=initimage, amount=amount, method=method, prompt=prompt)
            # display(outimage)
            # output.append({ "image": base64EncodeImage(outimage) })

        return jsonify(output)


DiffusersView.register(app)