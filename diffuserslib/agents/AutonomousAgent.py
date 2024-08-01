from .Agent import Agent
from diffuserslib.functional import FunctionalNode

class AutonomousAgent(Agent):

    def __init__(self, name:str, workflow:FunctionalNode):
        super().__init__(name)
        self.workflow = workflow
        
