from dataclasses import dataclass, asdict, field
from typing import Tuple, List
from ..TrainingParameters import TrainingParameters
import json

@dataclass
class LoraTrainingParameters(TrainingParameters):
    
    instancePrompt:str = ""

    priorPreservation:bool = False
    rank:int = 4                      # The dimension of the LoRA update matrices.
    maxGradientNorm:float = 1.0       # The maximum norm of the gradient.

    priorPreservation:bool = False
    classDir:str = ""
    classPrompt:str = ""
    numClassImages:int = 0
    priorLossWeight:float = 1.0      # The weight of the prior preservation loss.

    trainTextEncoder:bool = False
    textEncoderWeightDecay:float = 1e-03