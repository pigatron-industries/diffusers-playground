from .Vector import Vector
from typing import Callable, List


class MovingBody:
    def __init__(self, position:Vector, velocity:Vector):
        self.position = position
        self.velocity = velocity
    
    def update_position(self, dt):
        self.position += self.velocity * dt
    
    def __repr__(self):
        return f"MovingBody(Position: {self.position}, Velocity: {self.velocity})"


MovingBodyFuncType = MovingBody | Callable[[], MovingBody]
MovingBodiesFuncType = List[MovingBody] | Callable[[], List[MovingBody]]
