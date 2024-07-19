from diffuserslib.functional.FunctionalNode import FunctionalNode, NodeParameter
from diffuserslib.functional.nodes.user.UserInputNode import UserInputNode
from diffuserslib.interface.WorkflowController import WorkflowController
from ..WorkflowComponents import WorkflowComponents
from nicegui import ui
from PIL import Image


default_output_width = 256


class RealtimeInterfaceComponents(WorkflowComponents):

    def __init__(self, controller:WorkflowController):
        super().__init__(controller)
        self.output = None
        self.timer = ui.timer(0.1, lambda: self.updateWorkflowOutput(), active=False)
        self.running = False
        self.builders = {}
        for name, builder in self.controller.builders.items():
            if(builder.realtime):
                self.builders[name] = builder.name
        self.builders = dict(sorted(self.builders.items(), key=lambda item: item[1]))


    def runWorkflow(self):
        self.controller.saveWorkflowParamsToHistory()
        if(self.controller.model.workflow is not None and not self.controller.model.workflow.inited):
            self.controller.model.workflow.reset()
        self.timer.activate()
        self.running = True
        self.status.refresh()

    def stopWorkflow(self):
        self.timer.deactivate()
        self.running = False
        self.status.refresh()


    def updateWorkflowOutput(self):
        if(self.controller.model.workflow is not None):
            self.controller.model.workflow.flush()
            output = self.controller.model.workflow()
            self.output = output
            if(isinstance(output, Image.Image)):
                self.output_control.style(replace = f"max-width:{output.width}px; max-height:{output.height}px;")
                self.output_control.set_source(output)
            else:
                self.output_label.set_text(f"Unsupported output type: {type(output)}")
                self.output_control.set_source(None)


    def resetWorkflow(self):
        if(self.controller.model.workflow is not None):
            self.controller.model.workflow.reset()
            self.outputs.refresh()
            self.status.refresh()

    
    @ui.refreshable
    def status(self):
        with ui.column().classes("gap-0").style("align-items:center;"):  #margin-top:1.3em;
            if(self.running):
                ui.label("Running").style("color: #ffcc00;")
            else:
                ui.label("Idle").style("color: #00ff00;")

    
    def buttons(self):
        ui.number(label="Batch Size").bind_value(self.controller.model, 'batch_size').bind_visibility_from(self.controller.model, 'run_type', value=1).style("width: 100px")
        ui.button('Run', on_click=lambda e: self.runWorkflow()).classes('align-middle')
        ui.button('Stop', on_click=lambda e: self.stopWorkflow()).classes('align-middle')
        ui.button('Reset', on_click=lambda e: self.resetWorkflow()).classes('align-middle')


    @ui.refreshable
    def controls(self):
        self.workflowSelect(self.builders, ["Image"])
        self.workflow_parameters()


    @ui.refreshable
    def outputs(self):
        self.rundata_controls = {}
        self.output_container = ui.column().classes('w-full')
        with self.output_container:
            with ui.card_section().classes('w-full').style("background-color:#2b323b; border-radius:8px;"):
                self.output_label = ui.label("")
                if(self.output is not None):
                    self.output_control = ui.interactive_image(self.output).style(replace = f"max-width:{self.output.width}px; max-height:{self.output.height}px;")
                else:
                    self.output_control = ui.interactive_image()