from nicegui import ui
from diffuserslib.interface_nicegui import *
from diffuserslib.init import initializeDiffusers

initializeDiffusers(configs=["config/config.yml", "config/local_config.yml"], 
                    modelconfigs=["config/modelconfig.yml", "config/local_modelconfig.yml"],
                    promptmods=["config/promptmods.yml", "config/local_promptmods.yml"])

ui.run(port=8070, dark=True, storage_secret='secret-key')