class Variable:
    """
    Base class for optimization variables.
    Each variable has a unique ID and an optional name.

    Attributes:
        id (int): A unique identifier for the variable.
        name (str): A human-readable name for the variable.
    """
    _next_id = 0
    def __init__(self, name: str = ""):
        """
        Args:
            name (str, optional): Name of the variable. If empty, a default name
                                  based on the class and ID is generated.
        """
        self.id = Variable._next_id; Variable._next_id += 1
        self.name = name if name else f"{self.__class__.__name__}_{self.id}"
    def __hash__(self):
        return self.id
    def __eq__(self, other):
        return isinstance(other, Variable) and self.id == other.id
    def __repr__(self):
        return f"{self.name}" 