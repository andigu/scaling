from enum import IntEnum, unique

@unique
class Pauli(IntEnum):
    X = 0
    Z = 1
    Y = 2 # Y is 2 because it's rarely used

    def __str__(self):
        return ['X', 'Z', 'Y'][self.value]
