import math
from typing import Callable, List

class Vector:
    def __init__(self, *coordinates):
        self.coordinates = coordinates

    def __repr__(self):
        return f"Vector({', '.join(map(str, self.coordinates))})"
    
    def __add__(self, other):
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same number of dimensions to be added")
        return Vector(*[x + y for x, y in zip(self.coordinates, other.coordinates)])
    
    def __sub__(self, other):
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same number of dimensions to be subtracted")
        return Vector(*[x - y for x, y in zip(self.coordinates, other.coordinates)])
    
    def __mul__(self, scalar):
        return Vector(*[x * scalar for x in self.coordinates])
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        if scalar != 0:
            return Vector(*[x / scalar for x in self.coordinates])
        else:
            raise ValueError("Division by zero")
    
    def __eq__(self, other):
        return self.coordinates == other.coordinates
    
    def magnitude(self):
        return math.sqrt(sum(x**2 for x in self.coordinates))
    
    def normalize(self):
        mag = self.magnitude()
        if mag != 0:
            return self / mag
        else:
            raise ValueError("Cannot normalize the zero vector")
        
    def getX(self):
        return self.coordinates[0]
    
    def getY(self):
        return self.coordinates[1]
        

VectorFuncType = Vector | Callable[[], Vector]
VectorsFuncType = List[Vector] | Callable[[], List[Vector]]