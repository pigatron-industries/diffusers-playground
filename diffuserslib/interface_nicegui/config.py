from diffuserslib.functional.FunctionalTyping import *
from diffuserslib.functional import *
from typing import List

selectable_nodes_config:List[FunctionalNode] = [
    RandomIntNode(min_max=MinMaxIntInputNode(), name="random_int"), 
    RandomFloatNode(min_max=MinMaxFloatInputNode(), name="random_float"),
]
