import random
from typing import Callable


class ColourPalette:

    @classmethod
    def fromGradient(cls, start_colour, end_colour, num_colours):
        colour_list = []
        for i in range(num_colours):
            r = int(start_colour[0] + (end_colour[0] - start_colour[0]) * i / num_colours)
            g = int(start_colour[1] + (end_colour[1] - start_colour[1]) * i / num_colours)
            b = int(start_colour[2] + (end_colour[2] - start_colour[2]) * i / num_colours)
            colour_list.append((r, g, b))
        return cls(colour_list)

    def __init__(self, colour_list):
        self.colour_list = colour_list

    def getColour(self, index):
        return self.colour_list[index]

    def getRandomColour(self):
        return random.choice(self.colour_list)

    
ColourPaletteFuncType = ColourPalette | Callable[[], ColourPalette]