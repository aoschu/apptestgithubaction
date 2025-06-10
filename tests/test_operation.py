import math
from src.math_operations import add, subtract, multiply, divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(-5, -5) == -10
    assert math.isclose(add(2.5, 3.5), 6.0)
    assert math.isclose(add(1.1, 2.2), 3.3)
    assert add(1000000, 2000000) == 3000000

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 0) == 0
    assert subtract(-1, -1) == 0
    assert subtract(10, 5) == 5
    assert math.isclose(subtract(2.5, 1.5), 1.0)
    assert math.isclose(subtract(3.3, 1.1), 2.2)
    assert subtract(1000000, 500000) == 500000

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-1, 1) == -1
    assert multiply(0, 5) == 0
    assert multiply(4, 0) == 0
    assert math.isclose(multiply(2.5, 4), 10.0)
    assert math.isclose(multiply(1.1, 2.0), 2.2)
    assert multiply(1000000, 2000) == 2000000000

def test_divide():
    assert divide(6, 3) == 2
    assert divide(-6, 2) == -3
    assert divide(0, 1) == 0
    assert math.isclose(divide(5, 2), 2.5)
    assert math.isclose(divide(7.5, 2.5), 3.0)
    
    try:
        divide(1, 0)
    except ValueError as e:
        assert str(e) == "Cannot divide by zero."
    
    try:
        divide(0, 0)
    except ValueError as e:
        assert str(e) == "Cannot divide by zero."
