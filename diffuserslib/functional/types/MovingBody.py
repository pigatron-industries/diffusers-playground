from .Vector import Vector
from typing import Callable, List


class MovingBody:
    def __init__(self, position:Vector, velocity:Vector):
        self.position = position
        self.velocity = velocity
    
    def update_position(self, dt):
        self.position = self.position + (self.velocity * dt)

        # bounce off boundary
        if (self.position.getX() < 0):
            self.velocity = Vector(abs(self.velocity.getX()), self.velocity.getY())
        elif (self.position.getX() > 1):
            self.velocity = Vector(-abs(self.velocity.getX()), self.velocity.getY())
        if (self.position.getY() < 0):
            self.velocity = Vector(self.velocity.getX(), abs(self.velocity.getY()))
        elif (self.position.getY() > 1):
            self.velocity = Vector(self.velocity.getX(), -abs(self.velocity.getY()))

    
    def __repr__(self):
        return f"MovingBody(Position: {self.position}, Velocity: {self.velocity})"


MovingBodyFuncType = MovingBody | Callable[[], MovingBody]
MovingBodiesFuncType = List[MovingBody] | Callable[[], List[MovingBody]]
