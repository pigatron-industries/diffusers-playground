from nicegui import ui
from diffuserslib.interface.batch.index import *
from diffuserslib.interface.converse.index import *
from diffuserslib.interface.realtime.index import *
from diffuserslib.interface.api import *
from diffuserslib.init import initializeDiffusers

initializeDiffusers(configs=["config/config.yml", "config/local_config.yml"], 
                    modelconfigs=["config/modelconfig.yml", "config/local_modelconfig.yml"],
                    promptmods=["config/promptmods.yml", "config/local_promptmods.yml"])

ui.run(port=8070, reload=True, dark=True, storage_secret='secret-key')