from .Controller import Controller
from diffuserslib.functional.WorkflowRunner import WorkflowRunData
from diffuserslib.functional.types import Video, Audio
from nicegui import ui, run
from nicegui.element import Element
from nicegui.elements.label import Label
from PIL import Image
from typing import List


default_output_width = 256


class BatchRunDataControls:

    def __init__(self, rundata:WorkflowRunData, runid:int, batchid:int, rundata_container:Element, controller:Controller):
        self.rundata = rundata
        self.runid = runid
        self.batchid = batchid
        self.controller = controller
        self.rundata_container = rundata_container
        self.output_control = None
        self.output_width = 0
        self.label_saved = None
        self.waiting_output = False
        self.expanded = False
        self.workflow_output_rundata()


    def showLabelSaved(self, filename:str):
        if(self.label_saved is not None):
            self.label_saved.set_text(f"Saved to {filename}")
            self.label_saved.set_visibility(True)


    def update(self):
        if(self.waiting_output == True and self.rundata_container is not None):
            if(self.rundata.output is not None or self.rundata.error is not None or self.output_control is None):
                # run finished / errored / haven't created control yet / output type changed (TODO)
                # completely remove and re-add output controls
                self.rundata_container.clear()
                with self.rundata_container:
                    self.workflow_output_rundata()
            else:
                # just update progress
                self.workflow_update_preview()


    def workflow_output_rundata(self):
        with ui.row().classes('w-full no-wrap'):
            if(self.rundata.error is not None):
                self.waiting_output = False
                self.output_control = ui.label(f"Error: {self.rundata.error}").style("color: #ff0000;")
            elif(self.rundata.output is None):
                self.waiting_output = True
                self.workflow_generating()
            else:
                self.waiting_output = False
                self.workflow_output_preview()
                
            if(self.rundata.output is not None):
                with ui.column().classes('w-full'):
                    with ui.row().style("margin-left:auto;"):
                        ui.button('Params', on_click=lambda e: self.workflow_params_dialog())
                        ui.button('Save', on_click=lambda e: self.saveOutput())
                        ui.button('Copy', on_click=lambda e: self.controller.copyOutput(self.runid))
                        ui.button('Remove', on_click=lambda e: self.removeOutput())
                    with ui.row().style("margin-left:auto;"):
                        self.label_saved = ui.label(f"Saved to {self.rundata.save_file}")
                        self.label_saved.set_visibility(self.rundata.save_file is not None)
                    with ui.row().style("margin-left:auto;"):
                        ui.label(f"Duration: {self.rundata.duration}")
            else:
                self.label_saved = None


    async def saveOutput(self):
        result = await run.io_bound(self.controller.saveOutput, self.runid)
        if (self.rundata.save_file is not None):
            self.showLabelSaved(self.rundata.save_file)


    def removeOutput(self):
        self.controller.removeRunData(self.batchid, self.runid)
        self.rundata_container.delete()


    def workflow_generating(self):
        with ui.column():
            ui.label("Generating...").style(replace= f"max-width:{default_output_width}px; min-width:{default_output_width}px;")
            ui.linear_progress(show_value=False).bind_value_from(self, 'progress')
            progress = self.controller.getProgress()
            if(progress is not None and progress.run_progress is not None and progress.run_progress.output is not None):
                self.workflow_output_preview()


    def workflow_output_preview(self):
        if(isinstance(self.rundata.output, Image.Image)):
            image = self.rundata.preview if self.rundata.preview is not None else self.rundata.output
            self.output_width = image.width
            self.output_control = ui.image(image).on('click', lambda e: self.workflow_output_dialog()).style(f"max-width:{default_output_width}px; min-width:{default_output_width}px;")
        elif(isinstance(self.rundata.output, List) and len(self.rundata.output) > 0 and isinstance(self.rundata.output[-1], Image.Image)):
            image = self.rundata.output[-1]
            self.output_width = image.width
            self.output_control = ui.image(image).style(f"max-width:{default_output_width}px; min-width:{default_output_width}px;")
        elif(isinstance(self.rundata.output, Video)):
            self.output_width = 512  #TODO get video width
            self.output_control = ui.video(self.rundata.output.file.name).style(replace= f"max-width:{default_output_width}px; min-width:{default_output_width}px;")
        elif(isinstance(self.rundata.output, Audio)):
            self.rundata.output.write()
            self.output_control = ui.audio(self.rundata.output.file.name).style(replace= f"max-width:{default_output_width}px; min-width:{default_output_width}px;")
        else:
            self.output_width = 0
            self.output_control = None
            ui.label("Output format not supported").style("color: #ff0000;")


    def workflow_update_preview(self):
        progress = self.controller.getProgress()
        if(progress is not None and progress.run_progress is not None and self.output_control is not None):
            output = progress.run_progress.output
            self.progress = progress.run_progress.progress
            if(isinstance(output, Image.Image) and isinstance(self.output_control, ui.image)):
                self.output_control.set_source(output) # type: ignore
            elif(isinstance(output, List) and len(output) > 0 and isinstance(output[-1], Image.Image)):
                self.output_control.set_source(output[-1]) # type: ignore


    def workflow_output_dialog(self):
        with ui.dialog(value=True):
            if(isinstance(self.rundata.output, Image.Image)):
                ui.image(self.rundata.output).style(f"min-width:{self.rundata.output.width}px; min-height:{self.rundata.output.height}px;")


    def workflow_params_dialog(self):
        with ui.dialog(value=True), ui.card().style('height: 100%; max-width: 1000px;'):
            with ui.column():
                ui.label(f"Run {self.runid}")
                with ui.row():
                    ui.label(f"Duration: {self.rundata.duration}")
                if self.rundata.params is not None:
                    for node_name, node_output in self.rundata.params.items():
                        with ui.row().classes('no-wrap'):
                            ui.label(f"{node_name}:").style("line-height: 1;")
                            if(isinstance(node_output, Image.Image)):
                                ui.image(node_output).style("min-width:128px; min-height:128px;")
                            else:
                                ui.label(str(node_output)).style("line-height: 1;")