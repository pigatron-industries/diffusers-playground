from ..ImageProcessor import ImageProcessor, ImageContext
from PIL import ImageDraw
from typing import List, Tuple, Callable, Dict, Any
from scipy.spatial import Voronoi
import math
import numpy as np
import itertools


def edge_power_distribution(a, size):
    power_distribution = np.random.power(a=a, size=size)
    for idx in itertools.product(*[range(s) for s in size]):
        if (np.random.random() < 0.5):
            power_distribution[idx] = 1 - power_distribution[idx]
    return power_distribution


class DrawVoronoiDiagramProcessor(ImageProcessor):
    def __init__(self, 
                 points: List[Tuple[float, float]] | Callable[[], List[Tuple[float, float]]],
                 outline: str | Callable[[], str] = "white", 
                 draw: Tuple[bool, bool, bool] | Callable[[], Tuple[bool, bool, bool]] = (True, True, True),  # (bounded lines, unbounded lines, points)
                 lineProbablity: float | Callable[[], float] = 1,
                 radius: float = 2):
        args = {
            "points": points,
            "outline": outline,
            "draw": draw,
            "radius": radius,
            "lineProbablity": lineProbablity
        }
        super().__init__(args)

    def process(self, args:Dict[str, Any], inputImages:List[ImageContext], outputImage:ImageContext) -> ImageContext:        
        points = np.array(args["points"])
        outline = args["outline"]
        drawBoundedLines = args["draw"][0]
        drawUnboundedLines = args["draw"][1]
        drawPoints = args["draw"][2]
        radius = args["radius"]
        lineProbablity = args["lineProbablity"]

        points = points * inputImages[0].size
        lines = self.getLines(points, boundedLines=drawBoundedLines, unboundedLines=drawUnboundedLines)
        image = inputImages[0].getFullImage()
        draw = ImageDraw.Draw(image)

        for line in lines:
            if (np.random.random() < lineProbablity):
                draw.line(line, fill=outline, width=1)

        if (drawPoints):
            for point in points:
                draw.ellipse((point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius), fill="white")
            
        outputImage.setFullImage(image)
        return outputImage
    

    def getLines(self, points, boundedLines:bool = True, unboundedLines:bool = True) -> List[Tuple[float, float, float, float]]:
        voronoi = Voronoi(points)
        centre = voronoi.points.mean(axis=0)
        lines = []

        for ridge_points, ridge_vertices in voronoi.ridge_dict.items():
            if boundedLines and -1 not in ridge_vertices:
                # finite ridge between two vertices
                vert1 = voronoi.vertices[ridge_vertices[0]]
                vert2 = voronoi.vertices[ridge_vertices[1]]
                line = (vert1[0], vert1[1], vert2[0], vert2[1])
                lines.append(line)
            elif unboundedLines:
                # infinite ridge
                vert = voronoi.vertices[ridge_vertices[1]]
                point1 = voronoi.points[ridge_points[0]]
                point2 = voronoi.points[ridge_points[1]]
                midpoint = (point2 + point1) * 0.5
                normal = (midpoint - vert) / np.linalg.norm(midpoint - vert)
                direction = midpoint - centre
                normal = np.array([math.copysign(normal[0], direction[0]), math.copysign(normal[1], direction[1])])

                endpoint = vert + (normal * 512) # make line go twice as far as midpoint
                line = (vert[0], vert[1], endpoint[0], endpoint[1])
                lines.append(line)

        return lines