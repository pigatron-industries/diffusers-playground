import ipywidgets as widgets
from IPython.display import display
from typing import List, Dict, Callable
from PIL import Image
import copy
import glob
import os
from .WidgetHelpers import *
from ..batch.argument import RandomImageArgument
from ..processing.ProcessingPipeline import ImageProcessorPipeline


INIT_IMAGE = "Init Image"
PREV_IMAGE = "Previous Image"
RANDOM_IMAGE = "Random"


class InitImageInterface:
    def __init__(self, interface, firstImage = False):
        self.interface = interface
        self.firstImage = firstImage
        controlnet_models = list(interface.pipelines.presets.getModelsByType("controlnet").keys())
        if (firstImage):
            controlnet_models = [INIT_IMAGE] + controlnet_models
        generation_pipeline_options = list(interface.generation_pipelines.keys())
        inputdirs_options = interface.input_dirs
        if (not firstImage):
            inputdirs_options = [PREV_IMAGE] + inputdirs_options
        
        self.model_dropdown = dropdown(interface, label="Control Model:", options=controlnet_models, value=INIT_IMAGE if firstImage else None)
        self.generation_dropdown = dropdown(interface, label="Generation:", options=generation_pipeline_options, value=None)
        self.input_source_dropdown = dropdown(interface, label="Input Source:", options=inputdirs_options, value=None)
        self.input_select_dropdown = dropdown(interface, label="Input Select:", options=[], value=None)
        self.preprocessor_dropdown = dropdown(interface, label="Preprocessor:", options=[None]+list(interface.preprocessing_pipelines.keys()), value=None)

    def display(self):
        display(self.model_dropdown,
                self.generation_dropdown,
                self.input_source_dropdown,
                self.input_select_dropdown,
                self.preprocessor_dropdown,
                widgets.HTML("<span>&nbsp;</span>"))
        
    def updateWidgets(self):
        if (self.input_source_dropdown.value is not None and self.input_source_dropdown.value != PREV_IMAGE):
            filepaths = glob.glob(f"{self.input_source_dropdown.value}/*.png") + glob.glob(f"{self.input_source_dropdown.value}/*.jpg")
            self.input_select_dropdown.options = [RANDOM_IMAGE] + [os.path.basename(x) for x in filepaths]
        else:
            self.input_select_dropdown.options = []
        
    def hide(self):
        self.model_dropdown.layout.display = 'none'
        self.generation_dropdown.layout.display = 'none'
        self.input_source_dropdown.layout.display = 'none'
        self.input_select_dropdown.layout.display = 'none'
        self.preprocessor_dropdown.layout.display = 'none'

    def show(self):
        self.model_dropdown.layout.display = 'block'
        self.generation_dropdown.layout.display = 'block'
        self.input_source_dropdown.layout.display = 'block'
        self.input_select_dropdown.layout.display = 'block'
        self.preprocessor_dropdown.layout.display = 'block'


    def getInitImage(self):
        if(self.input_select_dropdown.value == RANDOM_IMAGE):
            return RandomImageArgument.fromDirectory(self.input_source_dropdown.value)
        else:
            return Image.open(self.input_source_dropdown.value + "/" + self.input_select_dropdown.value)
        

    def createGenerationPipeline(self, prevImageFunc:Callable[[], (Image.Image|None)]|Image.Image|None = None) -> ImageProcessorPipeline:
        pipeline = self.interface.generation_pipelines[self.generation_dropdown.value]
        pipeline = copy.deepcopy(pipeline)
        if(self.preprocessor_dropdown.value is not None):
            preprocessor = self.interface.preprocessing_pipelines[self.preprocessor_dropdown.value]
            pipeline.addTask(preprocessor())
        if(pipeline.hasPlaceholder("image")):
            if(self.input_source_dropdown.value != PREV_IMAGE):
                pipeline.setPlaceholder("image", self.getInitImage())
            elif (prevImageFunc is not None):
                pipeline.setPlaceholder("image", prevImageFunc)
        if(pipeline.hasPlaceholder("size")):
            pipeline.setPlaceholder("size", (self.interface.width_slider.value, self.interface.height_slider.value))
        return pipeline