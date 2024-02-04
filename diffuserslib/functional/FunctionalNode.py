from typing import Dict, Any
from ..batch import evaluateArguments


class FunctionalNode:
    def __init__(self, args:Dict[str, Any]):
        self.args = args

    def __call__(self) -> Any:
        args = evaluateArguments(self.args)
        return self.process(**args)
    
    def process(self, **kwargs):
        raise Exception("Not implemented")