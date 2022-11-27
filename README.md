# Stable Diffusion Playground

## To install and import in google colab


```
if not os.path.exists(f'/content/diffusers-playground'):
    !git clone https://github.com/pigatron-industries/diffusers-playground.git
else:
    %cd /content/diffusers-playground
    !git pull
    %cd /content/

!pip install -r diffusers-playground/requirements.txt


import importlib  
diffuserslib = importlib.import_module("diffusers-playground.diffuserslib")

```

