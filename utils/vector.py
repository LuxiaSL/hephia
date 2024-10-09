# utils/vector.py

class Vector2D:
    """
    Simple 2D vector class for position and movement calculations.
    """

    def __init__(self, x=0.0, y=0.0):
        """
        Initializes a Vector2D instance.

        Args:
            x (float): X-coordinate.
            y (float): Y-coordinate.
        """
        self.x = x
        self.y = y

    # Arithmetic operations
    def __add__(self, other):
        """Adds two vectors."""
        return Vector2D(self.x + other.x, self.y + other.y)

    def __iadd__(self, other):
        """In-place addition of two vectors."""
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other):
        """Subtracts two vectors."""
        return Vector2D(self.x - other.x, self.y - other.y)

    def __isub__(self, other):
        """In-place subtraction of two vectors."""
        self.x -= other.x
        self.y -= other.y
        return self

    def __mul__(self, scalar):
        """Multiplies vector by a scalar."""
        return Vector2D(self.x * scalar, self.y * scalar)

    def __imul__(self, scalar):
        """In-place multiplication of vector by a scalar."""
        self.x *= scalar
        self.y *= scalar
        return self
