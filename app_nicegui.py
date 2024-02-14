from nicegui import ui
from diffuserslib.interface_nicegui import *
from diffuserslib.init import initializeDiffusers

initializeDiffusers(configs=["local_config.yml", "config.yml"], modelconfigs=["local_modelconfig.yml", "modelconfig.yml"])

ui.run(port=8070, dark=True, storage_secret='secret-key')