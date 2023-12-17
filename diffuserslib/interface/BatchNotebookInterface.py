from ..batch import BatchRunner
from ..batch.argument import RandomNumberArgument, RandomImageArgument, RandomPromptProcessor
from ..inference import DiffusersPipelines
from ..inference.GenerationParameters import GenerationParameters, ControlImageParameters, ModelParameters, LoraParameters, ControlImageType
from ..processing import *
from ..processing.ImageProcessorPipeline import ImageProcessorPipeline
from ..processing.processors.filters import *
from ..processing.processors.transformers import *
from .InitImageInterface import *
from .LoraInterface import *
from .WidgetHelpers import *
import ipywidgets as widgets
import pickle 
import os
import functools
import multiprocessing
import time
from threading import Thread
from collections import OrderedDict
from typing import List, Dict, Callable
from functools import partial
from IPython.display import display, clear_output
from PIL import Image


DEFAULT_PREPROCESSORS = {
    'canny edge detection': CannyEdgeProcessor,
    'HED edge detection': HEDEdgeDetectionProcessor,
    'PIDI edge detection': PIDIEdgeDetectionProcessor,
    'MLSD straight line detection': MLSDStraightLineDetectionProcessor,
    'pose detection': PoseDetectionProcessor,
    'depth estimation': DepthEstimationProcessor,
    'normal estimation': NormalEstimationProcessor,
    'segmentation': SegmentationProcessor,
    'content shuffle': ContentShuffleProcessor,
    'monochrome': partial(SaturationProcessor, saturation=-1),
    'blur': partial(GaussianBlurProcessor, radius=2),
    'noise': partial(GaussianNoiseProcessor, sigma=10),
}


class OutputItem:
    def __init__(self, args:Dict, image:Image.Image|None = None):
        self.output:widgets.Output = widgets.Output()
        self.args = args
        self.image = image
        self.preserveprompt_checkbox = widgets.Checkbox(description="Preserve previous prompt", value=True)
        self.preservecontrol_checkbox = widgets.Checkbox(description="Preserve control images", value=False)
        self.preserveprompt_checkbox.layout.width = "300px"
        self.preservecontrol_checkbox.layout.width = "300px"

    def display(self, args:Dict, image:Image.Image, saveClick, refineClick, removeClick):
        with self.output:
            self.image = image
            self.args = args
            saveBtn = widgets.Button(description="Save")
            saveBtn.on_click(saveClick)
            refineBtn = widgets.Button(description="Refine")
            refineBtn.on_click(refineClick)
            refine_box = widgets.HBox([refineBtn, self.preserveprompt_checkbox, self.preservecontrol_checkbox])
            removeBtn = widgets.Button(description="Remove")
            removeBtn.on_click(removeClick)
            display(image, saveBtn, refine_box, removeBtn)


class OutputList:
    def __init__(self):
        self.output:widgets.Output = widgets.Output()
        self.outputs:List[OutputItem] = []

    def addOutput(self, args) -> Tuple[int, OutputItem]:
        item = OutputItem(args)
        self.outputs.append(item)
        with self.output:
            display(item.output)
        return (len(self.outputs) - 1, item)
    
    def getOutput(self, index) -> OutputItem:
        return self.outputs[index]
    
    def display(self, index, args, image, saveClick, refineClick, removeClick):
        self.outputs[index].display(args, image, functools.partial(saveClick, index), functools.partial(refineClick, index), functools.partial(removeClick, index))
    
    def removeOutput(self, index):
        self.outputs[index].output.clear_output()
    
    def clear(self):
        self.outputs = []
        self.output.clear_output()



class BatchNotebookInterface:
    def __init__(self, pipelines:DiffusersPipelines, output_dir:str, modifier_dict=None, save_file:str='batch_params.pkl', 
                 generation_pipelines:Dict[str, ImageProcessorPipeline]={}, 
                 preprocessing_pipelines:Dict[str, ImageProcessor]=DEFAULT_PREPROCESSORS, 
                 input_dirs:List[str]=[]):
        self.pipelines = pipelines
        self.output_dir = output_dir
        self.input_dirs = input_dirs
        self.modifier_dict = modifier_dict
        self.save_file = save_file
        self.generation_pipelines = generation_pipelines
        self.preprocessing_pipelines = preprocessing_pipelines

        #  Init images
        self.initimages_num = intSlider(self, label='Input Images:', value=0, min=0, max=4, step=1)
        self.initimage_widgets:List[InitImageInterface] = []
        self.initimage_widgets.append(InitImageInterface(self, firstImage=True))
        self.initimage_widgets.append(InitImageInterface(self))
        self.initimage_widgets.append(InitImageInterface(self))
        self.initimage_widgets.append(InitImageInterface(self))

        # Model options
        self.basemodel_dropdown = dropdown(self, label="Base Model:", options=["sd_1_5", "sd_2_1", "sdxl_1_0"], value=None)
        self.model_dropdown = dropdown(self, label="Model:", options=[], value=None)
        self.mergemodel_dropdown = dropdown(self, label="Model Merge:", options=[], value=None)
        self.mergeweight_slider = floatSlider(self, label='Merge Weight:', value=0.5, min=0, max=1, step=0.01)
        self.lora_num = intSlider(self, label='LORAs:', value=0, min=0, max=4, step=1)
        self.lora_widgets = []
        self.lora_widgets.append(LoraInterface(self))
        self.lora_widgets.append(LoraInterface(self))
        self.lora_widgets.append(LoraInterface(self))
        self.lora_widgets.append(LoraInterface(self))

        # Generation options
        self.prompt_text = textarea(self, label="Prompt:", value="")
        self.shuffle_checkbox = checkbox(self, label="Shuffle", value=False)
        self.negprompt_text = textarea(self, label="Neg Prompt:", value="")
        self.width_slider = intSlider(self, label='Width:', value=512, min=256, max=1536, step=64)
        self.height_slider = intSlider(self, label='Height:', value=768, min=256, max=1536, step=64)
        self.scale_slider = floatSlider(self, label='Guidance:', value=9, min=1, max=20, step=0.1)
        self.steps_slider = intSlider(self, label='Steps:', value=40, min=5, max=100, step=5)
        self.strength_slider = floatSlider(self, label='Strength:', value=0.5, min=0, max=1, step=0.01)
        self.scheduler_dropdown = dropdown(self, label="Sampler:", options=['DDIMScheduler', 'DPMSolverMultistepScheduler', 
                                                                           'EulerAncestralDiscreteScheduler', 'EulerDiscreteScheduler', 'LCMScheduler',
                                                                           'LMSDiscreteScheduler', 'UniPCMultistepScheduler'], value="EulerDiscreteScheduler")
        self.seed_text = intText(self, label='Seed:', value=None)
        self.batchsize_slider = intSlider(self, label='Batch:', value=10, min=1, max=100, step=1)

        self.run_button = widgets.Button(description="Run")
        self.clear_button = widgets.Button(description="Clear")
        self.output:OutputList = OutputList()

        self.run_button.on_click(self._runClick)
        self.clear_button.on_click(self._clearClick)

        html = widgets.HTML('''<style>
                        .widget-label { min-width: 20ex !important; }
                    </style>''')

        display(html,
                self.initimages_num)
        for initimage_w in self.initimage_widgets:
            initimage_w.display()
        display(widgets.HTML("<span>&nbsp;</span>"),
                self.basemodel_dropdown,
                self.model_dropdown, 
                self.mergemodel_dropdown,
                self.mergeweight_slider,
                self.lora_num)
        for lora_w in self.lora_widgets:
            lora_w.display()
        display(widgets.HTML("<span>&nbsp;</span>"),
                self.prompt_text, 
                self.shuffle_checkbox,
                self.negprompt_text, 
                self.width_slider, 
                self.height_slider, 
                self.scale_slider,
                self.steps_slider,
                self.strength_slider,
                self.scheduler_dropdown,
                self.seed_text,
                self.batchsize_slider,
                widgets.HTML("<span>&nbsp;</span>"),
                self.run_button,
                self.clear_button,
                self.output.output
        )
        self.loadParams()
        self.batchQueue:List[BatchRunner] = []
        # self.batchThread = multiprocessing.Process(target=self.batchThreadLoop)
        # self.batchThread.start()


    def updateWidgets(self):
        models = self.pipelines.presets.getModelsByTypeAndBase("generate", self.basemodel_dropdown.value)
        self.model_dropdown.options = list(models.keys())
        self.mergemodel_dropdown.options = [None] + list(models.keys())

        for i, initimage_w in enumerate(self.initimage_widgets):
            if(i < self.initimages_num.value):
                initimage_w.show()
            else:
                initimage_w.hide()

        if(self.mergemodel_dropdown.value is not None):
            self.mergeweight_slider.layout.display = 'flex'
        else:
            self.mergeweight_slider.layout.display = 'none'

        for i, lora_w in enumerate(self.lora_widgets):
            if(i < self.lora_num.value):
                lora_w.show()
                if(self.model_dropdown.value is not None):
                    lora_w.lora_dropdown.options = [None] + self.pipelines.getLORAList(self.model_dropdown.value)
            else:
                lora_w.hide()

        if(self.initimages_num.value > 0 and self.initimage_widgets[0].model_dropdown.value == INIT_IMAGE):
            self.steps_slider.layout.display = 'none'
        else:
            self.steps_slider.layout.display = 'flex'

        for initimage_w in self.initimage_widgets:
            initimage_w.updateWidgets()

    
    def onChange(self, change):
        if (change['type'] == 'change' and change['name'] == 'value'):
            self.updateWidgets()


    def getParams(self):
        params = OrderedDict()

        params['basemodel'] = self.basemodel_dropdown.value
        if(self.mergemodel_dropdown.value is None):
            params['models'] = [self.model_dropdown.value]
        else:
            params['models'] = [self.model_dropdown.value, self.mergemodel_dropdown.value]
        params['model_weight'] = self.mergeweight_slider.value
        params['init_prompt'] = self.prompt_text.value
        params['shuffle'] = self.shuffle_checkbox.value
        params['prompt'] = RandomPromptProcessor(modifier_dict = self.modifier_dict, 
                                                 wildcard_dict = self.pipelines.getEmbeddingTokens(self.basemodel_dropdown.value), 
                                                 prompt = str(self.prompt_text.value), 
                                                 shuffle = bool(self.shuffle_checkbox.value))
        params['negprompt'] = self.negprompt_text.value
        params['width'] = self.width_slider.value
        params['height'] = self.height_slider.value
        params['scale'] = self.scale_slider.value
        params['scheduler'] = self.scheduler_dropdown.value
        params['batch'] = self.batchsize_slider.value
        params['initimages_num'] = self.initimages_num.value

        prevImageFunc = None
        for i, initimage_w in enumerate(self.initimage_widgets):
            if(i >= params['initimages_num']):
                break
            params[f'initimage{i}_model'] = initimage_w.model_dropdown.value
            params[f'initimage{i}_generation'] = initimage_w.generation_dropdown.value
            params[f'initimage{i}_input_source'] = initimage_w.input_source_dropdown.value
            params[f'initimage{i}_input_select'] = initimage_w.input_select_dropdown.value
            params[f'initimage{i}_preprocessor'] = initimage_w.preprocessor_dropdown.value
            pipeline = initimage_w.createGenerationPipeline(prevImageFunc)
            prevImageFunc = pipeline.getLastOutput

            if(initimage_w.model_dropdown.value == INIT_IMAGE):
                params['initimage'] = pipeline
            else:
                if('controlimage' not in params):
                    params['controlimage'] = []
                if('controlmodel' not in params):
                    params['controlmodel'] = []
                params['controlimage'].append(pipeline)
                params['controlmodel'].append(initimage_w.model_dropdown.value)

        params['lora_num'] = self.lora_num.value
        for i, lora_w in enumerate(self.lora_widgets):
            if(i >= self.lora_num.value):
                break
            params[f'lora{i}_lora'] = lora_w.lora_dropdown.value
            params[f'lora{i}_loraweight'] = lora_w.loraweight_text.value
            if('loranames' not in params):
                params['loranames'] = []
            if('loraweights' not in params):
                params['loraweights'] = []
            params['loranames'].append(lora_w.lora_dropdown.value)
            params['loraweights'].append(lora_w.loraweight_text.value)

        params['strength'] = self.strength_slider.value
        params['steps'] = self.steps_slider.value

        if(self.seed_text.value > 0):
            params['seed'] = self.seed_text.value
        else:
            params['seed'] = RandomNumberArgument(0, 4294967295)

        return params
    

    def setParams(self, params):
        try:
            self.basemodel_dropdown.value = params.get('basemodel', 'sd_1_5')

            self.initimages_num.value = params.get('initimages_num', 0)
            for i, initimage_w in enumerate(self.initimage_widgets):
                initimage_w.model_dropdown.value = params.get(f'initimage{i}_model', None)
                initimage_w.generation_dropdown.value = params.get(f'initimage{i}_generation', None)
                initimage_w.input_source_dropdown.value = params.get(f'initimage{i}_input_source', None)
                initimage_w.input_select_dropdown.value = params.get(f'initimage{i}_input_select', None)
                initimage_w.preprocessor_dropdown.value = params.get(f'initimage{i}_preprocessor', None)

            model = params.get('models', None)
            self.model_dropdown.value = model[0]
            if (len(model) > 1):
                self.mergemodel_dropdown.value = model[1]
            self.mergeweight_slider.value = params.get('model_weight', 1)

            self.lora_num.value = params.get('lora_num', 0)
            for i, lora_w in enumerate(self.lora_widgets):
                lora_w.lora_dropdown.value = params.get(f'lora{i}_lora', None)
                lora_w.loraweight_text.value = params.get(f'lora{i}_loraweight', 1)

            self.prompt_text.value = params.get('init_prompt', '')
            self.negprompt_text.value = params.get('negprompt', '')
            self.width_slider.value = params.get('width', 512)
            self.height_slider.value = params.get('height', 512)
            self.scale_slider.value = params.get('scale', 9.0)
            self.steps_slider.value = params.get('steps', 40)
            self.strength_slider.value = params.get('strength', 0.5)
            self.scheduler_dropdown.value = params.get('scheduler', 'EulerDiscreteScheduler')
            self.batchsize_slider.value = params.get('batch', 10)
        except Exception as e:
            print(e)
            print("Error loading params")
        self.updateWidgets()


    def saveParams(self):
        params = self.getParams()
        filtered_params = { key: value for key, value in params.items() if key not in ['initimage', 'controlimage'] }
        with open(self.save_file, 'wb') as f:
            pickle.dump(filtered_params, f)
        return params


    def loadParams(self):
        if(os.path.isfile(self.save_file)):
            with open(self.save_file, 'rb') as f:
                params = pickle.load(f)
                self.setParams(params)
        self.updateWidgets()


    def batchThreadLoop(self):
        while(True):
            if(len(self.batchQueue) > 0):
                batch = self.batchQueue.pop(0)
                batch.run()
            time.sleep(0.1)


    def startGenerationCallback(self, args:Dict) -> Tuple[int, widgets.Output]:
        index, outputItem = self.output.addOutput(args)
        return index, outputItem.output


    def endGenerationCallback(self, output_index:int, args:Dict, image:Image.Image):
        self.output.display(output_index, args, image, self._saveClick, self._refineClick, self._removeClick)


    def _runClick(self, b):
        with self.output.output:
            self.run()


    def _clearClick(self, b):
        self.output.clear()
        self.images = []
        self.args = []


    def _saveClick(self, index, b):
        outputItem = self.output.getOutput(index)
        with outputItem.output:
            args = outputItem.args
            image = outputItem.image
            if(image is not None):
                timestamp = args['timestamp']
                image_filename = f"{self.output_dir}/txt2img_{timestamp}.png"
                info_filename = f"{self.output_dir}/txt2img_{timestamp}.txt"
                image.save(image_filename)
                self._saveArgs(args, info_filename)
                print("Saved to: " + image_filename)


    def _saveArgs(self, args, file):
        with open(file, 'w') as file:
            description = ""
            for arg in args.keys():
                value = args[arg]
                if (isinstance(value, str) or isinstance(value, int) or isinstance(value, float)):
                    description += f"{arg}: {value}\n"
                    file.write(f"{arg}: {value}\n")
                elif (isinstance(value, Image.Image) and hasattr(value, 'filename')):
                    description += f"{arg}: {getattr(value, 'filename')}\n"
                    file.write(f"{arg}: {getattr(value, 'filename')}\n")
                elif (isinstance(value, list)):
                    if (all(isinstance(item, str) for item in value) or all(isinstance(item, int) for item in value) or all(isinstance(item, float) for item in value)):
                        description += f"{arg}: {value}\n"
                        file.write(f"{arg}: {value}\n")


    def _removeClick(self, index, b):
        with self.output.output:
            self.output.removeOutput(index)


    def _refineClick(self, index, b):
        with self.output.output:
            outputItem = self.output.getOutput(index)
            self.refine(outputItem.image, outputItem.args, outputItem.preserveprompt_checkbox.value, outputItem.preservecontrol_checkbox.value)


    def createBatch(self, params):
        batch = BatchRunner(self.generate, self.output_dir, self.startGenerationCallback, self.endGenerationCallback)
        return batch
    

    def generate(self, prompt, negprompt, width, height, steps, scale, scheduler, seed, models, strength, initimage=None, 
                 controlimage:List[Image.Image]|None = None, controlmodel:List[str]|None = None, 
                 loranames=None, loraweights=None, model_weight=1.0, **kwargs):
        controlimageparams = []
        if(initimage is not None):
            controlimageparams.append(ControlImageParameters(image=initimage, model=ControlImageType.IMAGETYPE_INITIMAGE))
        if(controlimage is not None and controlmodel is not None):
            for i in range(0, len(controlimage)):
                modelid = controlmodel[i].split(":")[1]
                controlimageparams.append(ControlImageParameters(image=controlimage[i], type=ControlImageType.IMAGETYPE_CONTROLIMAGE, model=modelid))

        loraparams = []
        if(loranames is not None and loraweights is not None):
            for i in range(0, len(loranames)):
                loraparams.append(LoraParameters(name=loranames[i], weight=loraweights[i]))

        modelparams = [ModelParameters(name=models[0], weight=1.0)]
        if(len(models) > 1):
            modelparams.append(ModelParameters(name=models[1], weight=model_weight))

        params = GenerationParameters(prompt=prompt, negprompt=negprompt, width=width, height=height, steps=steps, cfgscale=scale, strength=strength, scheduler=scheduler, seed=seed, 
                                      models=modelparams, loras=loraparams, controlimages=controlimageparams)
        return self.pipelines.generate(params)


    def run(self):
        params = self.saveParams()
        batch = self.createBatch(params)
        batch.appendBatchArguments(params, params['batch'])
        batch.run()


    def refine(self, image, args, preserveprompt:bool=False, preservecontrol:bool=False):
        params = self.getParams()

        # modify params for refinement
        params['initimage'] = image
        if(preserveprompt):
            params['prompt'] = args['prompt']
        if(preservecontrol and 'controlimage' in args):
            params['controlimage'] = args['controlimage']
        elif('controlimage' in args):
            del params['controlimage']
            del params['controlmodel']

        batch = self.createBatch(params)
        batch.appendBatchArguments(params, params['batch'])
        # self.batchQueue.append(batch)
        batch.run()
