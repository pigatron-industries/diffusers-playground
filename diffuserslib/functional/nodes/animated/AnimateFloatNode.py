from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *


class LinearInterpolation:
    def __init__(self, min_value:float, max_value:float):
        self.min_value = min_value
        self.max_value = max_value

    def interpolate(self, phase:float) -> float:
        return self.min_value + (self.max_value - self.min_value) * phase



class AnimateFloatNode(FunctionalNode):
    def __init__(self, 
                 interpolator:LinearInterpolation = LinearInterpolation(0.0, 1.0),
                 init_phase:FloatFuncType = 0.0,
                 dt:FloatFuncType = 0.01,
                 name:str = "animate_float"):
        super().__init__(name)
        self.interpolator = interpolator
        self.addInitParam("init_phase", init_phase, float)
        self.addParam("dt", dt, float)
        self.reset()


    def init(self, init_phase:float):
        self.phase = init_phase


    def process(self, dt:float) -> float:
        self.phase += dt
        if(self.phase > 1.0):
            self.phase -= 1.0
        return self.interpolator.interpolate(self.phase)
    