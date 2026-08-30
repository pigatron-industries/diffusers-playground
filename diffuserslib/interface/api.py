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
from diffuserslib.functional.nodes.image.diffusers.ImageDiffusionNode import ImageDiffusionNode
from diffuserslib.functional import WorkflowRunner
from diffuserslib.interface.WorkflowController import WorkflowController
from diffuserslib.functional_workflows.image.ImageDiffusionWorkflow import ImageDiffusionWorkflow


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
    def loras(model: str | None = None, base: str | None = None):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("Pipelines not initialized")
        pipelines = DiffusersPipelines.pipelines
        if base:
            try:
                return pipelines.getLORAsByBase(base)
            except Exception:
                return []
        if model:
            try:
                return pipelines.getLORAList(model)
            except Exception:
                return []

        raise Exception("Must provide either 'model' or 'base' parameter")
    
    
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
        try:
            print('=== generate (workflow) ===')
            RestApi.prescaleBefore(params)

            if WorkflowRunner.workflowrunner is None:
                raise Exception("WorkflowRunner not initialized")

            # Build the ImageDiffusion workflow and populate user-input nodes
            if WorkflowRunner.workflowrunner is None:
                raise Exception("WorkflowRunner not initialized")

            # TODO this could be made more generic
            controller = WorkflowController.getInstance()
            controller.loadWorkflow('ImageDiffusionWorkflow')
            workflow_node = controller.model.workflow

            # helper to safely get node by type or name, maybe a load workflow and set all named parameters function
            def get_node_by_type(node, t):
                try:
                    return node.getNodeByType(t)
                except Exception:
                    return None

            def get_node_by_name(node, name):
                try:
                    return node.getNodeByName(name)
                except Exception:
                    return None

            # Diffusion model input
            dm_node = get_node_by_type(workflow_node, __import__('diffuserslib.functional.nodes.image.diffusers.user.DiffusionModelUserInputNode', fromlist=['DiffusionModelUserInputNode']).DiffusionModelUserInputNode)
            if dm_node is not None and len(params.models) > 0:
                base = params.models[0].base if hasattr(params.models[0], 'base') else None
                models_list = [(m.name, m.weight) for m in params.models]
                dm_node.setValue((base, models_list))

            # LORA node
            try:
                from diffuserslib.functional.nodes.image.diffusers.user.LORAModelUserInputNode import LORAModelUserInputNode
                lora_node = get_node_by_type(workflow_node, LORAModelUserInputNode)
                if lora_node is not None:
                    lora_node.setValue([(l.name, l.weight) for l in params.loras] if params.loras else [])
            except Exception:
                lora_node = None

            # other simple inputs
            n = get_node_by_name(workflow_node, 'prompt')
            if n is not None:
                n.setValue(params.prompt)
            n = get_node_by_name(workflow_node, 'negprompt')
            if n is not None:
                n.setValue(params.negprompt)
            n = get_node_by_name(workflow_node, 'seed')
            if n is not None:
                n.setValue(params.seed)
            n = get_node_by_name(workflow_node, 'steps')
            if n is not None:
                n.setValue(params.steps)
            n = get_node_by_name(workflow_node, 'cfgscale')
            if n is not None:
                n.setValue(params.cfgscale)
            n = get_node_by_name(workflow_node, 'scheduler')
            if n is not None:
                n.setValue(params.scheduler)
            n = get_node_by_name(workflow_node, 'clipskip')
            if n is not None:
                n.setValue(params.clipskip)
            # sigmas: try to match dict option values to params.sigmas
            n = get_node_by_name(workflow_node, 'sigmas')
            if n is not None:
                # n.getSelectedOption not guaranteed; find matching key
                for key, val in getattr(n, 'dict', {}).items():
                    if val == params.sigmas:
                        n.setValue(key)
                        break
                else:
                    # fallback for None
                    if params.sigmas is None and 'None' in getattr(n, 'dict', {}):
                        n.setValue('None')

            # Save workflow params to history (UI does this before running)
            controller.saveWorkflowParamsToHistory()

            # Enqueue as a workflow batch and wait for completion
            batchid = WorkflowRunner.workflowrunner.run(workflow_node, batch_size=int(params.batch))

            # Poll until all runs in the batch have completed
            import time
            while True:
                batch = WorkflowRunner.workflowrunner.getBatch(batchid)
                if batch is None:
                    # no batch found (should not normally happen) - break
                    break
                # check if we've collected all run results
                if len(batch.rundata) >= batch.batch_size:
                    all_done = True
                    for rd in batch.rundata.values():
                        if rd.error is None and rd.end_time is None:
                            all_done = False
                            break
                    if all_done:
                        break
                time.sleep(0.25)

            # Collect outputs
            outputimages = []
            batch = WorkflowRunner.workflowrunner.getBatch(batchid)
            if batch is None:
                raise Exception("Batch data missing after run")

            for rd in batch.rundata.values():
                if rd.error is not None:
                    RestApi.job.status = { "status":"error", "action":"generate", "error":str(rd.error) }
                    raise rd.error
                outimage = rd.output
                outimage = RestApi.prescaleAfter([outimage], params)[0]
                outputimages.append({ "seed": None, "image": base64EncodeImage(outimage) })

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
                usedseed = 0
                if (params.tilemethod=="auto"):
                    # TODO make this use a full workflow, will need to convert params into conditioning param nodes
                    # TODO support use of differential img2img by passing a mask tile
                    tilesize_calc = TileSizeCalculatorNode(image_size = (params.width, params.height), overlap = params.tileoverlap)()
                    outimage, usedseed = ImageDiffusionTiledNode.tiledGeneration(params=params, tilewidth=tilesize_calc[0], tileheight=tilesize_calc[1], overlap=params.tileoverlap)
                elif (params.tilemethod=="singlepass"):
                    outimage, usedseed = tiledProcessorCentred(tileprocessor=tiledGeneration, pipelines=DiffusersPipelines.pipelines, params=params, tilewidth=params.tilewidth, tileheight=params.tileheight, 
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