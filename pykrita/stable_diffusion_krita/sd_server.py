from .sd_config import *
from .sd_common import *

from krita import *
import urllib.request
import http.client
import time
import json


class SDParameters:
    "This is Stable Diffusion Parameter Class"     
    model = None
    control_model = None
    prompt = ""
    negprompt = ""
    steps = 0
    seed = 0
    num =0
    strength=1.0
    scale=0.5
    seedList =["","","",""]
    imageDialog = None
    regenerate = False
    images64 = []
    controlmodels = []
    maskImage64=""
    scheduler="DPMSolverMultistepScheduler"
    inpaint_mask_blur=4
    inpaint_mask_content="latent noise" 
    action="txt2img"
    strength = 1 
    upscale_amount = 1
    upscale_method = None
    tile_method = None
    tile_width = None
    tile_height = None
    tile_overlap = None
    tile_alignmentx = None
    tile_alignmenty = None
    process = None


INIT_IMAGE_MODEL = "Init Image"


def createRequest(action, params):
    if (not params.seed): 
        seed=None
    else: 
        seed=int(params.seed)
    method = None
    if(params.upscale_method is not None):
        method = params.upscale_method
    elif(params.tile_method is not None):
        method = params.tile_method

    # move init image to front of image list
    controlmodels = params.controlmodels
    if(params.images64 is not None and len(params.images64) > 0):
        controlmodels = params.controlmodels.copy()
        if(controlmodels.count(INIT_IMAGE_MODEL) > 1):
            raise Exception("Multiple init images found in image list")
        if(action == "inpaint" and INIT_IMAGE_MODEL not in controlmodels):
            raise Exception("init image not found in image list")
        try:
            initIndex = controlmodels.index(INIT_IMAGE_MODEL)
            params.images64.insert(0, params.images64.pop(initIndex))
            controlmodels.pop(initIndex)
        except ValueError:
            pass

    request = { 
        'model': params.model,
        'prompt': params.prompt,
        'negprompt': params.negprompt,
        'controlimages': params.images64,
        'controlmodels': controlmodels,
        'steps':params.steps,
        'scheduler':params.scheduler,
        'mask_blur': SDConfig.inpaint_mask_blur,
        'use_gfpgan': False,
        'batch': params.num,
        'scale': params.scale,
        'strength': params.strength,
        'seed':seed,
        'height':SDConfig.height,
        'width':SDConfig.width,
        'method': method,
        'amount': params.upscale_amount,
        'upscale_overlap':64,
        'inpaint_full_res':True,
        'inpainting_mask_invert': 0,
        'tilewidth': params.tile_width,
        'tileheight': params.tile_height,
        'tileoverlap': params.tile_overlap,
        'tilealignmentx': params.tile_alignmentx,
        'tilealignmenty': params.tile_alignmenty,
        'process': params.process
    }    
    return request


def getServerDataAsync(action, params):
    if(action == "img2img" and INIT_IMAGE_MODEL not in params.controlmodels):
        action = "txt2img"

    data = createRequest(action, params)
    reqData = json.dumps(data).encode("utf-8")
    endpoint=SDConfig.url
    endpoint=endpoint.strip("/")
    endpoint+="/api/"
    asyncEndpoint = endpoint+"async"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }    
    try:
        # do a 'ping' to check server is  running first
        req = urllib.request.Request(endpoint, None, headers, method="GET")
        with urllib.request.urlopen(req) as f:
            res = f.read()
        # make initial request to start async job
        req = urllib.request.Request(asyncEndpoint+"/"+action, reqData, headers, method="POST")
        with urllib.request.urlopen(req) as f:
            res = f.read()

        progress = QProgressDialog("Running SD...", "Cancel", 0, 100)
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        progress.show()
        progress.forceShow()
        i = 0

        while (True):
            time.sleep(5)
            # poll get enpoint for response
            req = urllib.request.Request(asyncEndpoint, None, headers, method="GET")
            with urllib.request.urlopen(req) as f:
                res = f.read()
            data = json.loads(res)

            i = i + 1
            done = data.get("done", 0)
            total = data.get("total", 1)
            if(done == 0):
                done = 1
            else:
                done = done*100 / total
            progress.setValue(done)
            progress.setLabelText(data.get("description", ""))
            progress.show()
            progress.forceShow()

            if(data["status"] == "finished"):
                return res
            elif(data["status"] == "error"):
                errorMessage("Job Error", "Reason: "+data["error"])
                return None

    except http.client.IncompleteRead as e:
        print("Incomplete Read Exception - better restart Colab or ")
        res = e.partial 
        return res           
    except Exception as e:
        error_message = traceback.format_exc() 
        errorMessage("Server Error", "Endpoint: "+endpoint+", Reason: "+error_message)        
        return None



def getServerData(action, params):
    data = createRequest(params)
    reqData = json.dumps(data).encode("utf-8")
    endpoint=SDConfig.url
    endpoint=endpoint.strip("/")
    endpoint+="/api/"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }    
    try:
        print("endpoint")
        print(endpoint)
        req = urllib.request.Request(endpoint, None, headers, method="GET") # do a 'ping' to check server is  running first
        with urllib.request.urlopen(req) as f:
            res = f.read()
        req = urllib.request.Request(endpoint+action, reqData, headers, method="POST")
        with urllib.request.urlopen(req) as f:
            res = f.read()
            return res
    except http.client.IncompleteRead as e:
        print("Incomplete Read Exception - better restart Colab or ")
        res = e.partial 
        return res           
    except Exception as e:
        error_message = traceback.format_exc() 
        errorMessage("Server Error","Endpoint: "+endpoint+", Reason: "+error_message)        
        return None

