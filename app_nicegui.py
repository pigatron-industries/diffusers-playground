from nicegui import ui
from diffuserslib.interface_nicegui import *
from diffuserslib.init import initializeDiffusers

initializeDiffusers(configs=["config.yml", "local_config.yml"], modelconfigs=["modelconfig.yml", "local_modelconfig.yml"])

ui.run(port=8070, dark=True, storage_secret='secret-key')