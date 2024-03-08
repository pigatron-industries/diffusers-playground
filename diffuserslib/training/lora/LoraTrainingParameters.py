from dataclasses import dataclass, asdict, field
from typing import Tuple, List
from ..TrainingParameters import TrainingParameters
import json

@dataclass
class LoraTrainingParameters(TrainingParameters):
    
    priorPreservation:bool = False
    rank:int = 4                      # The dimension of the LoRA update matrices.

    instancePrompt:str = ""

    priorPreservation:bool = False
    classDir:str = ""
    classPrompt:str = ""
    numClassImages:int = 0

    trainTextEncoder:bool = False
    textEncoderWeightDecay:float = 1e-03