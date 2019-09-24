from enum import Enum


class Chamber(Enum):
    """
    Represents which chamber of the bicameral body
    """
    SENATE = 1  # Upper
    HOUSE = 2  # Lower
