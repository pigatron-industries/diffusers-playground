from .Interpolation import *
import sys
from ..ImageUtils import cv2ToPil, pilToCv2
from PIL import Image
import cv2

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class Transform:
    def __init__(self, length, interpolation:InterpolationFunction=LinearInterpolation()):
        self.length = length
        self.interpolation = interpolation
        self.frame = 0
        self.calcTimings()

    def calcTimings(self):
        step_size = 1 / self.length
        frame_positions = [i*step_size for i in range(self.length+1)]
        self.frametimings = [self.interpolation(x) for x in frame_positions]
        self.frametimings_diff = [self.frametimings[i+1] - self.frametimings[i] for i in range(len(self.frametimings) - 1)]

    def __call__(self, image):
        outimage = self.transform(image)
        self.frame = self.frame + 1
        return outimage


class ZoomTransform(Transform):
    def __init__(self, length, interpolation:InterpolationFunction=LinearInterpolation(), xcentre=0, ycentre=0, zoom=1, **kwargs):
        super().__init__(length, interpolation)
        self.xcentre = xcentre
        self.ycentre = ycentre
        self.zoom = zoom

    def transform(self, image):
        image = pilToCv2(image)
        height, width = image.shape[:2]
        frame_zoom = self.zoom ** (self.frametimings_diff[self.frame])
        frame_xcentre = (width/2)*self.xcentre + (width/2)
        frame_ycentre = (height/2)*self.ycentre + (height/2)
        rotation_matrix = cv2.getRotationMatrix2D((frame_xcentre, frame_ycentre), 0, frame_zoom)
        zoomed_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        return cv2ToPil(zoomed_image)


class RotateTransform(Transform):
    def __init__(self, length, interpolation:InterpolationFunction=LinearInterpolation(), xcentre=0, ycentre=0, angle=0, zoom=1, **kwargs):
        super().__init__(length, interpolation)
        self.xcentre = xcentre
        self.ycentre = ycentre
        self.zoom = zoom
        self.angle = angle

    def transform(self, image):
        image = pilToCv2(image)
        height, width = image.shape[:2]
        frame_zoom = self.zoom ** (self.frametimings_diff[self.frame])
        frame_angle = self.angle * self.frametimings_diff[self.frame]
        frame_xcentre = (width/2)*self.xcentre + (width/2)
        frame_ycentre = (height/2)*self.ycentre + (height/2)
        rotation_matrix = cv2.getRotationMatrix2D((frame_xcentre, frame_ycentre), frame_angle, frame_zoom)
        zoomed_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        return cv2ToPil(zoomed_image)
