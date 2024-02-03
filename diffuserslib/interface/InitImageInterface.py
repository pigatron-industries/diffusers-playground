import ipywidgets as widgets
from IPython.display import display
from typing import List, Dict, Callable
from PIL import Image
import copy
import glob
import os
from .WidgetHelpers import *
from ..batch.argument import RandomImageArgument
from ..processing.ImageProcessorPipeline import ImageProcessorPipeline
from ..FileUtils import getFileList


INIT_IMAGE = "Init Image"
PREV_IMAGE = "Previous Image"
RANDOM_IMAGE = "Random"


class InitImageInterface:
    def __init__(self, interface, firstImage = False):
        self.visible = False
        self.interface = interface
        self.firstImage = firstImage
        generation_pipeline_options = list(interface.generation_pipelines.keys())
        inputdirs_options = interface.input_dirs
        if (not firstImage):
            inputdirs_options = [PREV_IMAGE] + inputdirs_options
        
        self.model_dropdown = dropdown(interface, label="Control Model:", options=[], value=None)
        self.scale_slider = floatSlider(interface, label='Weight:', value=0.8, min=0, max=1, step=0.01)
        self.generation_dropdown = dropdown(interface, label="Generation:", options=generation_pipeline_options, value=None)
        self.input_source_dropdown = dropdown(interface, label="Input Source:", options=inputdirs_options, value=None)
        self.input_select_dropdown = dropdown(interface, label="Input Select:", options=[], value=None)
        self.preprocessor_dropdown = dropdown(interface, label="Preprocessor:", options=[None]+list(interface.preprocessing_pipelines.keys()), value=None)

    def display(self):
        display(self.model_dropdown,
                self.scale_slider,
                self.generation_dropdown,
                self.input_source_dropdown,
                self.input_select_dropdown,
                self.preprocessor_dropdown,
                widgets.HTML("<span>&nbsp;</span>"))
        
    def updateWidgets(self):
        controlmodels = self.interface.pipelines.presets.getModelsByTypeAndBase("controlimage", self.interface.basemodel_dropdown.value)
        modelnames = [f'{model.modeltype}:{model.modelid}' for modelid, model in controlmodels.items()]
        if (self.firstImage):
            modelnames = [INIT_IMAGE] + modelnames
        self.model_dropdown.options = modelnames

        if (self.input_source_dropdown.value is not None and self.input_source_dropdown.value != PREV_IMAGE):
            filelist = getFileList(directory, patterns = ["*.jpg", "*.jpeg", "*.png"], recursive = recursive)
            self.input_select_dropdown.options = [RANDOM_IMAGE] + filelist
        else:
            self.input_select_dropdown.options = []

        
    def hide(self):
        self.visible = True
        self.model_dropdown.layout.display = 'none'
        self.scale_slider.layout.display = 'none'
        self.generation_dropdown.layout.display = 'none'
        self.input_source_dropdown.layout.display = 'none'
        self.input_select_dropdown.layout.display = 'none'
        self.preprocessor_dropdown.layout.display = 'none'

    def show(self):
        self.visible = False
        self.model_dropdown.layout.display = 'flex'
        self.scale_slider.layout.display = 'flex'
        self.generation_dropdown.layout.display = 'flex'
        self.preprocessor_dropdown.layout.display = 'flex'
        if (self.generation_dropdown.value is not None):
            pipeline = self.interface.generation_pipelines[self.generation_dropdown.value]
            if(pipeline.hasPlaceholder("image")):
                self.input_source_dropdown.layout.display = 'flex'
                self.input_select_dropdown.layout.display = 'flex'
            else:
                self.input_source_dropdown.layout.display = 'none'
                self.input_select_dropdown.layout.display = 'none'


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