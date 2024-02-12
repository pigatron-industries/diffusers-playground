from typing import Callable, Tuple, List, NewType
from PIL import Image

# Python type definitions for node inputs
StringFuncType = str | Callable[[], str]
IntFuncType = int | Callable[[], int]
FloatFuncType = float | Callable[[], float]
BoolFuncType = bool | Callable[[], bool]

StringsFuncType = List[str] | Callable[[], List[str]]
IntsFuncType = List[int] | Callable[[], List[int]]
FloatsFuncType = List[float] | Callable[[], List[float]]
BoolsFuncType = List[bool] | Callable[[], List[bool]]

SizeType = Tuple[int, int]
SizeFuncType = SizeType | Callable[[], SizeType]

MinMaxIntType = Tuple[int, int]
MinMaxIntFuncType = MinMaxIntType | Callable[[], MinMaxIntType]
MinMaxFloatType = Tuple[float, float]
MinMaxFloatFuncType = MinMaxFloatType | Callable[[], MinMaxFloatType]

Point2DType = Tuple[float, float]
Point2DFuncType = Point2DType | Callable[[], Point2DType]
Points2DType = List[Point2DType]
Points2DFuncType = Points2DType | Callable[[], Points2DType]

ImageFuncType = Image.Image | Callable[[], Image.Image]

ColourType = Tuple[int, int, int] | str
ColourFuncType = ColourType | Callable[[], ColourType]


def ConstrainedFloat(min_value: float, max_value: float):
    def validator(value):
        if not isinstance(value, int):
            raise TypeError("Value must be a float")
        if value < min_value or value > max_value:
            raise ValueError(f"Value must be between {min_value} and {max_value}")
        return value
    return NewType('ConstrainedFloat', float)

ProbabilityType = ConstrainedFloat(0.0, 1.0)
ProbabilityFuncType = ProbabilityType | Callable[[], ProbabilityType]

