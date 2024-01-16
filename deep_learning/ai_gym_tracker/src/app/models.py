"""Base model class for all counters in the application."""
from abc import ABC, abstractmethod
from typing import List
from src.app import mp_pose

# Create a Counter base class
class Counter(ABC):
    """
    A base class for Counter objects.

    Attributes
    ----------
    counter : int
        The current count.
    state : str
        The current state of the counter.
    landmarks : list
        The landmarks of the body.
    """
    def __init__(self, state = None, output = None):
        self.counter = 0
        self.landmarks = None
        self.state = state
        self.output = output

    @abstractmethod
    def make_calculations(self):
        """
        Takes care of the calculations needed to count.
        """
        ...

    @abstractmethod
    def count(self):
        """
        Takes care of counting logic like incrementing, decrementing, and resetting the counter.
        """
        ...

    def run(self, landmarks: List[mp_pose.PoseLandmark]) -> None:
        """
        Args:
            landmarks (List[mp_pose.PoseLandmark]): List of landmarks.
        """
        self.landmarks = landmarks
        self.make_calculations()
        self.count()

    def increment(self):
        """Increments the counter by 1."""
        self.counter += 1

    def decrement(self):
        """Decrements the counter by 1."""
        self.counter -= 1

    def reset(self):
        """Resets the counter to 0."""
        self.counter = 0

    def __repr__(self):
        if self.output:
            return str(self.output)
        else:
            return str(f"Counter: {str(self.counter)} State: {str(self.state)}")
