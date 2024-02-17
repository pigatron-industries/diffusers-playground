from diffuserslib.functional.FunctionalTyping import *
from diffuserslib.functional import *
from typing import List

selectable_nodes_config:List[FunctionalNode] = [
    RandomIntNode(min_max = MinMaxIntInputNode(), name = "Random Integer"), 
    RandomFloatNode(min_max = MinMaxFloatInputNode(), name = "Random Float"),
    RandomImageNode(paths = FileSelectInputNode(), name = "Random image")
]
