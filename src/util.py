from enum import Enum


class ValueEqEnum(Enum):
    # This is a bit of a hack, but it's useful for development with live-reloading since it
    # decides equality based on the value rather than the object identity.
    def __eq__(self, other: object) -> bool:
        return self.value == other.value

    # Below are some useful methods for usage with argparse
    # __str__ and from_string for argparse from here https://stackoverflow.com/questions/43968006/support-for-enum-arguments-in-argparse
    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)
        
    @classmethod
    def from_string(cls, s):
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError()
