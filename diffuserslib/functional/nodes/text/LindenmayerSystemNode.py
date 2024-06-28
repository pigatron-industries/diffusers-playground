from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import ImageDraw, Image
from typing import Dict


class LindenmayerSystemNode(FunctionalNode):
    def __init__(self, 
                 rules:List[str],
                 axiom:str,
                 iterations:int,
                 name:str = "lsystem"):
        super().__init__(name)
        self.addParam("rules", rules, List[str])
        self.addParam("axiom", axiom, str)
        self.addParam("iterations", iterations, int)


    def process(self, rules:List[str], axiom:str, iterations:int) -> str:
        derived = self.derivation(axiom, rules, iterations)
        return derived


    def derivation(self, axiom:str, rules:List[str], iterations:int) -> str:
        system_rules = {}
        for rule in rules:
            key, value = rule.split("->")
            system_rules[key] = value

        derived = [axiom]
        for _ in range(iterations):
            seq = derived[-1]
            new_seq = [self.rule(system_rules, char) for char in seq]
            derived.append("".join(new_seq))
        return axiom
    

    def rule(self, system_rules:Dict[str, str], char:str) -> str:
        if char in system_rules:
            return system_rules[char]
        return char
