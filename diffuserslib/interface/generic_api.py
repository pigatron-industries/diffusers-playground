from nicegui import app
from fastapi import HTTPException
from threading import Thread, Lock
from typing import Any, Dict
from pydantic import BaseModel, Field
from PIL import Image
from io import BytesIO
import base64
import datetime
import os
import uuid

from diffuserslib.functional import FunctionalNode, NodeParameter, UserInputNode, WorkflowRunner, Video, Audio
from diffuserslib.functional.nodes.user.FileUploadInputNode import FileUploadInputNode
from diffuserslib.ImageUtils import base64EncodeImage
from diffuserslib.interface.WorkflowController import WorkflowController
from .api import RestApi


class GenericWorkflowRequest(BaseModel):
    workflow:str
    params:Dict[str, Any] = Field(default_factory=dict)
    batch_size:int = 1


class GenericJob:
    """A single queued generic-workflow job with its own status/result."""

    def __init__(self, jobid:str):
        self.id = jobid
        self.status:Dict[str, Any] = { "status":"queued", "action":"generic" }
        self.created_at = datetime.datetime.now().isoformat()
        self.thread:Thread|None = None


class GenericApi:
    """
    A generic REST endpoint that can run *any* registered workflow by name.

    Unlike the focused endpoints in api.py (which hard-code how a specific
    workflow's input nodes are populated), this endpoint takes the workflow name
    plus a flat map of user-input-node values and applies them generically.

    The param keys use the same dotted "path" format that the UI history /
    saveWorkflowParamsToHistory uses, with the trailing ".value" stripped,
    e.g. "size", "prompt", "models". Use the discovery endpoints below to
    inspect the exact keys and the value each input node expects.

    The workflow is queued exactly like the generate endpoint: it is submitted
    to the shared WorkflowRunner batch queue and polled to completion.

    Async runs are tracked as independent jobs (see GenericJob): each POST to
    /api/generic/async/run gets its own job id, so multiple workflows can be
    queued at once and polled individually via /api/generic/async/{job_id}.
    """

    # Registry of async jobs: jobid -> GenericJob. Dicts preserve insertion
    # order, which we use to prune the oldest finished jobs first.
    jobs:Dict[str, GenericJob] = {}
    jobs_lock:Lock = Lock()
    MAX_JOBS:int = 128

    #================= DISCOVERY =================
    @staticmethod
    @app.get("/api/generic/workflows")
    def listWorkflows():
        controller = WorkflowController.getInstance()
        workflows = []
        for name, builder in controller.builders.items():
            workflows.append({
                "name": name,
                "display_name": builder.name,
                "output_type": builder.type.__name__ if builder.type is not None else None,
                "subworkflow": builder.subworkflow,
            })
        return workflows


    @staticmethod
    @app.get("/api/generic/workflows/{workflow}/params")
    def workflowParams(workflow:str):
        """ Returns the user-input-node keys, the node type, and a sample value for each. """
        wf = GenericApi._buildWorkflow(workflow)
        params = {}
        def visitor(param, parents):
            paramstring = '.'.join([parent.name if isinstance(parent, NodeParameter) else str(parent) for parent in parents])
            if isinstance(param.value, UserInputNode):
                params[paramstring] = {
                    "node_type": param.value.__class__.__name__,
                    "value": GenericApi._safeGetValue(param.value),
                }
        wf.visitParams(visitor)
        return params


    #================= RUN =================
    @staticmethod
    @app.post("/api/generic/run")
    def run(request:GenericWorkflowRequest):
        return GenericApi.genericRun(request)


    @staticmethod
    @app.post("/api/generic/async/run")
    def runAsync(request:GenericWorkflowRequest):
        job = GenericJob(uuid.uuid4().hex)
        with GenericApi.jobs_lock:
            GenericApi.jobs[job.id] = job
            # Bound memory: drop the oldest finished jobs once over the cap.
            if len(GenericApi.jobs) > GenericApi.MAX_JOBS:
                for oldid in list(GenericApi.jobs):
                    if len(GenericApi.jobs) <= GenericApi.MAX_JOBS:
                        break
                    if GenericApi.jobs[oldid].status.get("status") in ("finished", "error"):
                        del GenericApi.jobs[oldid]
        job.thread = Thread(target=GenericApi.genericRun, args=(request, job))
        job.thread.start()
        return { "job_id": job.id, **job.status }


    @staticmethod
    @app.get("/api/generic/async/{job_id}")
    def getJob(job_id:str):
        with GenericApi.jobs_lock:
            job = GenericApi.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job id '{job_id}'")
        return { "job_id": job.id, **job.status }


    @staticmethod
    def genericRun(request:GenericWorkflowRequest, job:GenericJob|None = None):
        def setStatus(status:Dict[str, Any]):
            if job is not None:
                job.status = status
            else:
                RestApi.job.status = status

        try:
            print('=== generic workflow run ===')
            if (WorkflowRunner.workflowrunner is None):
                raise Exception("WorkflowRunner not initialized")

            setStatus({ "status":"running", "action":"generic", "workflow": request.workflow })

            workflow = GenericApi._buildWorkflow(request.workflow)
            applied, unmatched = GenericApi._applyParams(workflow, request.params)

            # Enqueue as a workflow batch and wait for completion (same as the generate endpoint)
            batchid = WorkflowRunner.workflowrunner.run(workflow, batch_size=int(request.batch_size))
            GenericApi._waitForBatch(batchid)

            batch = WorkflowRunner.workflowrunner.getBatch(batchid)
            if batch is None:
                raise Exception("Batch data missing after run")

            outputs = []
            for runid, rd in batch.rundata.items():
                if rd.error is not None:
                    setStatus({ "status":"error", "action":"generic", "error":str(rd.error) })
                    raise rd.error
                outputs.append(GenericApi._serializeOutput(runid, rd.output))

            status = {
                "status":"finished",
                "action":"generic",
                "workflow": request.workflow,
                "outputs": outputs,
                "applied_params": list(applied),
                "unmatched_params": unmatched,
            }
            setStatus(status)
            return status

        except Exception as e:
            setStatus({ "status":"error", "action":"generic", "error":str(e) })
            raise e


    #================= HELPERS =================
    @staticmethod
    def _buildWorkflow(workflow_name:str) -> FunctionalNode:
        """Build a fresh instance of a registered workflow from its builder."""
        controller = WorkflowController.getInstance()
        if workflow_name not in controller.builders:
            available = sorted(controller.builders.keys())
            raise Exception(f"Unknown workflow '{workflow_name}'. Available: {available}")
        built = controller.builders[workflow_name].build()
        if isinstance(built, tuple):
            return built[0]
        return built


    @staticmethod
    def _applyParams(workflow:FunctionalNode, params:Dict[str, Any]) -> tuple:
        """
        Apply a flat map of {paramPath: value} to the workflow's user-input nodes.
        Returns (applied_keys:set, unmatched_keys:list).
        """
        applied = set()
        def visitor(param, parents):
            paramstring = '.'.join([parent.name if isinstance(parent, NodeParameter) else str(parent) for parent in parents])
            if isinstance(param.value, UserInputNode):
                key = paramstring
                if key in params:
                    GenericApi._setNodeValue(param.value, params[key])
                    applied.add(key)
        workflow.visitParams(visitor)
        unmatched = [key for key in params.keys() if key not in applied]
        return applied, unmatched


    @staticmethod
    def _setNodeValue(node:UserInputNode, value):
        """
        Set a node's value. Most input nodes expose setValue(). Image/file upload
        nodes are the exception: their setValue() only records a filename, but the
        workflow reads the image via processValue() (from the content list populated
        by GUI uploads). For those, accept a base64-encoded image or a file path and
        populate the content list so the value is actually used.
        """
        if isinstance(node, FileUploadInputNode):
            if value is None:
                node.setValue(None)
                return
            image = GenericApi._decodeImage(value)
            node.content = [image]
            node.filename = "api-upload"
            return
        node.setValue(value)


    @staticmethod
    def _decodeImage(value) -> Image.Image:
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, str):
            # base64 (optionally with a data: URL prefix)
            b64 = value.split(",", 1)[1] if value.startswith("data:") else value
            try:
                return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            except Exception:
                pass
            # fall back to a local file path
            if os.path.isfile(b64):
                return Image.open(b64).convert("RGB")
        raise Exception("Image input must be a base64-encoded image (optionally a data: URL) or a local file path")


    @staticmethod
    def _waitForBatch(batchid:int):
        import time
        while True:
            batch = WorkflowRunner.workflowrunner.getBatch(batchid)
            if batch is None:
                break
            if len(batch.rundata) >= batch.batch_size:
                all_done = True
                for rd in batch.rundata.values():
                    if rd.error is None and rd.end_time is None:
                        all_done = False
                        break
                if all_done:
                    break
            time.sleep(0.25)


    @staticmethod
    def _safeGetValue(node:UserInputNode):
        try:
            return node.getValue()
        except Exception:
            return None


    @staticmethod
    def _serializeOutput(runid, output):
        runner = WorkflowRunner.workflowrunner
        if output is None:
            return { "type":"None", "value":None }
        if isinstance(output, Image.Image):
            return { "type":"Image", "image": base64EncodeImage(output) }
        if isinstance(output, Video):
            runner.save(runid)
            return { "type":"Video", "file": runner.rundata[runid].save_file }
        if isinstance(output, Audio):
            runner.save(runid)
            return { "type":"Audio", "file": runner.rundata[runid].save_file }
        if isinstance(output, str):
            return { "type":"str", "value": output }
        # Fallback for any other output type
        return { "type": type(output).__name__, "value": str(output) }
