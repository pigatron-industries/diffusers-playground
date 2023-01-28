from .sd_config import *
from .sd_common import *

from krita import *
import urllib.request
import http.client
import time
import json



def getServerDataAsync(action, reqData):
    endpoint=SDConfig.url
    endpoint=endpoint.strip("/")
    endpoint+="/api/"
    asyncEndpoint = endpoint+"async"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }    
    try:
        print("endpoint")
        print(endpoint)
        # do a 'ping' to check server is  running first
        req = urllib.request.Request(endpoint, None, headers, method="GET")
        with urllib.request.urlopen(req) as f:
            res = f.read()
        # make initial request to start async job
        req = urllib.request.Request(asyncEndpoint+"/"+action, reqData, headers, method="POST")
        with urllib.request.urlopen(req) as f:
            res = f.read()
        while (True):
            time.sleep(5)
            # poll get enpoint for response
            req = urllib.request.Request(asyncEndpoint, None, headers, method="GET")
            with urllib.request.urlopen(req) as f:
                res = f.read()
            data = json.loads(res)
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



def getServerData(action, reqData):
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

