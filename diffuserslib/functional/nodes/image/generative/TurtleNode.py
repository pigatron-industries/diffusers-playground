from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from PIL import Image, ImageDraw
import math
from typing import Tuple
from dataclasses import dataclass
import copy 


@dataclass
class TurtleState:
    pos: Tuple[float, float]
    heading: float


class Turtle:
    def __init__(self, draw:ImageDraw.ImageDraw, pos:Tuple[int, int]=(0,0), heading:float=0):
        self.draw = draw
        self.state = TurtleState(pos, heading)
        self.stack:List[TurtleState] = []
        

    def forward(self, distance:float):
        self.pos = (self.state.pos[0] + math.cos(math.radians(self.state.heading)) * distance, 
                   self.state.pos[1] + math.sin(math.radians(self.state.heading)) * distance)


    def line(self, distance:float, colour:ColourType):
        new_pos = (self.state.pos[0] + math.cos(math.radians(self.state.heading)) * distance, 
                   self.state.pos[1] + math.sin(math.radians(self.state.heading)) * distance)
        self.draw.line([self.state.pos, new_pos], fill=colour)
        self.state.pos = new_pos


    def turn(self, angle:float):
        self.state.heading += angle


    def push(self):
        self.stack.append(copy.deepcopy(self.state))


    def pop(self):
        self.state = self.stack.pop()
    


class TurtleNode(FunctionalNode):
    def __init__(self, 
                 image:ImageFuncType,
                 name:str = "turtle",
                 instructions:StringFuncType = "F",
                 colour:ColourFuncType = "white",
                 init_position:FloatTupleFuncType = (0.5, 0.5),
                 init_heading:FloatFuncType = 270,
                 init_turning_angle:FloatFuncType = 90,
                 init_line_length:IntFuncType = 10,
                 line_length_factor:FloatFuncType = 1.0):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("instructions", instructions, str)
        self.addParam("colour", colour, ColourType)
        self.addParam("init_position", init_position, float)
        self.addParam("init_heading", init_heading, float)
        self.addParam("init_turning_angle", init_turning_angle, float)
        self.addParam("init_line_length", init_line_length, int)
        self.addParam("line_length_factor", line_length_factor, float)


    def process(self, image:Image.Image, instructions:str, colour:ColourType,
                init_position:Tuple[float, float], init_heading:float,
                init_turning_angle:float, init_line_length:int, line_length_factor:float) -> Image.Image: 
        line_length = float(init_line_length)
        turning_angle = init_turning_angle
        draw = ImageDraw.Draw(image)
        turtle = Turtle(draw, (int(image.width*init_position[0]), int(image.height*(1-init_position[1]))), init_heading)

        print(instructions)

        for instruction in instructions:
            if instruction == "F":
                turtle.line(line_length, colour)
            elif instruction == "f":
                turtle.forward(line_length)
            elif instruction == "+":
                turtle.turn(turning_angle)
            elif instruction == "-":
                turtle.turn(-turning_angle)
            elif instruction == "[":
                turtle.push()   
            elif instruction == "]":
                turtle.pop()
            elif instruction == ">":
                line_length *= line_length_factor
            elif instruction == "<":
                line_length /= line_length_factor

        return image
    