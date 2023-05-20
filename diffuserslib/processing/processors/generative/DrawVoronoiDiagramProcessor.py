from ...ProcessingPipeline import ImageProcessor, ImageContext
from ....batch import evaluateArguments

from PIL import ImageDraw
from typing import List, Tuple, Callable
from scipy.spatial import Voronoi
import math
import numpy as np


class DrawVoronoiDiagramProcessor(ImageProcessor):
    def __init__(self, 
                 points: List[Tuple[float, float]] | Callable[[], List[Tuple[float, float]]],
                 outline: str | Callable[[], str] = "white", 
                 draw: Tuple[bool, bool] | Callable[[], Tuple[bool, bool]] = (True, True),  # (lines, points)
                 radius: float = 2):
        self.args = {
            "points": points,
            "outline": outline,
            "draw": draw,
            "radius": radius
        }

    def __call__(self, context:ImageContext):
        if (context.image is None):
            raise ValueError("ImageContext must have an image to draw on")
        
        args = self.evaluateArguments(context=context)
        points = np.array(args["points"])
        outline = args["outline"]
        drawLines = args["draw"][0]
        drawPoints = args["draw"][1]
        radius = args["radius"]

        points = points * context.size
        lines = self.getLines(points)
        draw = ImageDraw.Draw(context.image)

        if (drawLines):
            for line in lines:
                draw.line(line, fill=outline, width=1)

        if (drawPoints):
            for point in points:
                draw.ellipse((point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius), fill="white")
            
        return context
    

    def getLines(self, points) -> List[Tuple[float, float, float, float]]:
        voronoi = Voronoi(points)
        centre = voronoi.points.mean(axis=0)
        lines = []

        for ridge_points, ridge_vertices in voronoi.ridge_dict.items():
            if -1 not in ridge_vertices:
                # finite ridge between two vertices
                vert1 = voronoi.vertices[ridge_vertices[0]]
                vert2 = voronoi.vertices[ridge_vertices[1]]
                line = (vert1[0], vert1[1], vert2[0], vert2[1])
                lines.append(line)
            else:
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