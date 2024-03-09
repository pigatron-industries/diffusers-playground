from dataclasses import dataclass, asdict, field
from typing import Tuple, List
from ..TrainingParameters import TrainingParameters
import json

@dataclass
class EmbeddingTrainingParameters(TrainingParameters):
    
    initializerToken: str = 'person'    # A token to use as initializer word.
    learnableProperty: str = 'object'   # Choose between 'object' and 'style' and 'subject_style'
    subject: str = ''                   # The subject of the example data, only used when `learnableProperty` is 'subject_style'.
    numVectors: int|None = None         # Number of vectors to train.


    def __init__(self, base:str, outputDirPrefix:str|None=None, **kwargs):
        super().__init__(base, outputDirPrefix, **kwargs)