# utils/helpers.py

def clamp(value, min_value, max_value):
    """
    Clamps the value between min_value and max_value.

    Args:
        value (float): The value to clamp.
        min_value (float): The minimum allowed value.
        max_value (float): The maximum allowed value.

    Returns:
        float: The clamped value.
    """
    return max(min_value, min(value, max_value))
