from dataclasses import dataclass, asdict, field
from typing import Tuple, List
from ..TrainingParameters import TrainingParameters
import json

@dataclass
class LoraTrainingParameters(TrainingParameters):
    
    priorPreservation:bool = False
    trainTextEncoder:bool = False
    rank:int = 4                      # The dimension of the LoRA update matrices.