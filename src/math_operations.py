def add(a,b):
    """Returns the sum of a and b."""
    return a + b
def subtract(a, b):
    """Returns the difference of a and b."""
    return a - b
def multiply(a, b):
    """Returns the product of a and b."""
    return a * b
def divide(a, b):
    """Returns the quotient of a and b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b
def power(base, exponent):
    """Returns base raised to the power of exponent."""
    return base ** exponent
def modulus(a, b):
    """Returns the remainder of a divided by b."""
    if b == 0:
        raise ValueError("Cannot perform modulus with zero.")
    return a % b
def square_root(a):
    """Returns the square root of a. Raises ValueError if a is negative."""
    if a < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return a ** 0.5
def absolute_value(a):
    """Returns the absolute value of a."""
    return abs(a)
def factorial(n):
    """Returns the factorial of n. Raises ValueError if n is negative."""
    if n < 0:
        raise ValueError("Cannot compute factorial of a negative number.")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
def gcd(a, b):
    """Returns the greatest common divisor of a and b."""
    while b:
        a, b = b, a % b
    return abs(a)
def lcm(a, b):
    """Returns the least common multiple of a and b."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)
def is_prime(n):
    """Returns True if n is a prime number, otherwise False."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
def fibonacci(n):
    """Returns the nth Fibonacci number. Raises ValueError if n is negative."""
    if n < 0:
        raise ValueError("Cannot compute Fibonacci of a negative number.")
    if n == 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
def nth_prime(n):
    """Returns the nth prime number. Raises ValueError if n is less than 1."""
    if n < 1:
        raise ValueError("n must be a positive integer.")
    count = 0
    num = 1
    while count < n:
        num += 1
        if is_prime(num):
            count += 1
    return num
def sum_of_squares(n):
    """Returns the sum of squares of the first n natural numbers. Raises ValueError if n is negative."""
    if n < 0:
        raise ValueError("Cannot compute sum of squares for a negative number.")
    return sum(i ** 2 for i in range(1, n + 1))
def product_of_squares(n):
    """Returns the product of squares of the first n natural numbers. Raises ValueError if n is negative."""
    if n < 0:
        raise ValueError("Cannot compute product of squares for a negative number.")
    product = 1
    for i in range(1, n + 1):
        product *= i ** 2
    return product
def arithmetic_mean(numbers):
    """Returns the arithmetic mean of a list of numbers. Raises ValueError if the list is empty."""
    if not numbers:
        raise ValueError("Cannot compute mean of an empty list.")
    return sum(numbers) / len(numbers)
def geometric_mean(numbers):
    """Returns the geometric mean of a list of numbers. Raises ValueError if the list is empty or contains non-positive numbers."""
    if not numbers:
        raise ValueError("Cannot compute geometric mean of an empty list.")
    product = 1
    count = 0
    for number in numbers:
        if number <= 0:
            raise ValueError("Geometric mean is undefined for non-positive numbers.")
        product *= number
        count += 1
    return product ** (1 / count) if count > 0 else 0