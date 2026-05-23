
# https://docs.python.org/3/library/typing.html

"""
# Warning: Expected type 'int', got 'str' instead (factor), TypeError: can't multiply sequence by non-int of type 'str'
def multiply_string(factor: int, string: str) -> str:
    return factor * string

x = multiply_string("world", "hello ")
print("Correct example: ", x)


# Warning: Expected type 'int', got 'str' instead (return)
def multiply_string(factor: int, string: str) -> int:
    return factor * string

x = multiply_string(2, "hello ")
print("Correct example: ", x)

"""


# Correct example
def multiply_string(factor: int, string: str) -> str:
    return factor * string

x = multiply_string(2, "hello ")
print("Correct example: ", x)


# Type Hinting - List alias
y: str = 'Lorem ipsum dolor sit amet'
y2: int = 1_000_000
y3: float = 0.5
y4: bool = True

print(y, y2, y3, y4)


# Type aliases FOR COLLECTION!

from typing import List, Tuple, Set, Dict, Union, Any, Optional

z1: List = [1, 2, 3]
z2: Tuple = (1, 2, 3)
z3: Set = {1, 2, 3}
z4: Dict = {'one': 1, 'two': 2, 'three': 3}

print(z1, z2, z3, z4)


# Precise Type
l1: List[int] = [1, 2, 3]
l2: List[Union[int, str]] = ['text', 1, 2,]
l3: List[Tuple[Any, Any]] = [('1', 'two'), (3, 4.0)]  # tuples with 2 elements

print(l1, l2, l3)

x1: List[int] = [1,2,3]
print(type(x1))


# Optional arguments - Optional[Type]` is shorter version of `Union[Type, None]
def multiply(a: int, b: int, c: Optional[int] = None) -> int:
   return a * b * c if c else a * b

print(multiply(5, 6))
print(multiply(5, 6, 3))


# Callable object

from typing import Callable

def do_something() -> int:
    return 2

fun: Callable = do_something

print(fun()) # call the function that is stored in the fun variable works like do_something()
print(do_something())


# Structure

from dataclasses import dataclass

# Create a Point class that stores two numbers: x and y
@dataclass

class Point:
    x: int = 0
    y: int = 0

point: Point = Point()

print(point)
print(f"Point: {point.x}, {point.y}")


