from .sd_config import *
from .sd_common import *
from .sd_parameters import *

from krita import * # type: ignore
import urllib.request
import http.client
import time
import json


def serverClipboardCopy(image:str):
    try:
        config = SDConfig()
        body = ClipboardContent(contenttype="image", content=base64EncodeImage(image)).toJson().encode("utf-8")
        endpoint = config.url.strip("/") + "/api/clipboard"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        print(endpoint)
        request = urllib.request.Request(endpoint, body, headers, method="POST")
        with urllib.request.urlopen(request) as f:
            response = f.read()
        print(response)
    except Exception as e:
        error_message = traceback.format_exc() 
        errorMessage("Server Error","Endpoint: "+endpoint+", Reason: "+error_message)


def serverClipboardPaste():
    config = SDConfig()
    endpoint = config.url.strip("/") + "/api/clipboard"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }    
    request = urllib.request.Request(endpoint, None, headers, method="GET")
    with urllib.request.urlopen(request) as f:
        response = f.read()
    data = json.loads(response)
    content = ClipboardContent.from_dict(data)
    return base64ToQImage(content.content)


def getServerDataAsync(action, params):
    config = SDConfig()
    reqData = params.toJson().encode("utf-8")
    endpoint=config.url.strip("/") + "/api/"
    asyncEndpoint = endpoint+"async"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }    

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
        try:
            time.sleep(5)
            # poll get endpoint for response
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
                done = int(done*100 / total)
            progress.setValue(done)
            progress.setLabelText(data.get("description", ""))
            progress.show()
            progress.forceShow()

            if(data["status"] == "finished"):
                return res
            elif(data["status"] == "error"):
                if not errorMessageRetry("Job Error", "Reason: "+data["error"]):
                    return None
        
        except Exception as e:
            error_message = traceback.format_exc() 
            if not errorMessageRetry("Server Error", "Endpoint: "+endpoint+", Reason: "+error_message):
                return None



def getServerData(action, params):
    reqData = params.toJson().encode("utf-8")
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

