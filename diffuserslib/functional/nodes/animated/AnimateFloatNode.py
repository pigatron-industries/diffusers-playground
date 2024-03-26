from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *



class AnimateFloatNode(FunctionalNode):
    def __init__(self, 
                 init_phase:FloatFuncType = 0.0,
                 dt:FloatFuncType = 0.01,
                 name:str = "animate_float"):
        super().__init__(name)
        self.addInitParam("init_phase", init_phase, float)
        self.addParam("dt", dt, float)
        self.reset()


    def init(self, init_phase:float):
        self.phase = init_phase


    def process(self, dt:float) -> float:
        self.phase += dt
        if(self.phase > 1.0):
            self.phase -= 1.0
        return self.phase
    


class AnimateFloatRampNode(AnimateFloatNode):
    def __init__(self, 
                 init_phase:FloatFuncType = 0.0,
                 dt:FloatFuncType = 0.01,
                 min_max:MinMaxFloatFuncType = (0.0, 1.0),
                 name:str = "ramp_float"):
        super().__init__(init_phase, dt, name)
        self.addParam("min_max", min_max, MinMaxFloatType)

    
    def process(self, dt:float, min_max:MinMaxFloatType) -> float:
        self.phase += dt
        if(self.phase > 1.0):
            self.phase -= 1.0
        return min_max[0] + (min_max[1] - min_max[0]) * self.phase