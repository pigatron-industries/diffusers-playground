from nicegui import ui
from diffuserslib.interface_nicegui import *
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines

DiffusersPipelines.pipelines = DiffusersPipelines(device = "mps", safety_checker=True, cache_dir=None)
DiffusersPipelines.pipelines.loadPresetFile("modelconfig.yml")

ui.run(port=8070, dark=True, storage_secret='secret-key')