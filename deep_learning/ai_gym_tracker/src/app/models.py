"""Base model class for all counters in the application."""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from src.app import mp_pose

# Create a Counter base class
class Counter(ABC):
    """
    A base class for Counter objects.

    Attrs:
    ------
    counter : int
        The current count.
    reps_per_set : Optional[int]
        The number of reps per set.
    num_sets : Optional[int]
        The number of sets.
    current_set : int
        The current set.
    reps_this_set : int
        The number of reps this set.
    landmarks : List[mp_pose.PoseLandmark]
        List of landmarks.
    state : str
        The current state.
    """
    def __init__(
        self,
        state: str = "start",
        output: Dict[str, str] = None,
        reps_per_set: Optional[int] = None,
        num_sets: Optional[int] = None,
        start_pose_image_path: str = ""
    ) -> None:
        """
        Args:
            state (str): The current state.
            output (Dict[str, str]): The output.
            reps_per_set (Optional[int]): The number of reps per set.
            num_sets (Optional[int]): The number of sets.
            start_pose_image_path (str): The path to the start pose image.
        """
        self.counter = 0
        self.counter_type = self.__class__.__name__
        self.reps_per_set = reps_per_set
        self.num_sets = num_sets
        self.current_set = 1
        self.reps_this_set = 0

        self.landmarks: List[mp_pose.PoseLandmark] = []
        self.state = state
        self.output = output or {}

        self.start_pose_image_path = start_pose_image_path

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

    def run(self, landmarks: List[mp_pose.PoseLandmark]) -> int:
        """
        Args:
            landmarks (List[mp_pose.PoseLandmark]): List of landmarks.
        """
        self.landmarks = landmarks
        self.make_calculations()
        self.count()
        return self.counter
    
    def is_finished(self) -> bool:
        """
        Returns:
            bool: True if the counter is finished, False otherwise.
        """
        return self.current_set > self.num_sets

    def increment(self):
        """Increments the counter by 1."""
        self.counter += 1

    def decrement(self):
        """Decrements the counter by 1."""
        self.counter -= 1

    def reset(self):
        """Resets the counter to 0."""
        self.counter = 0

    def get_info(self):
        """
        Returns:
            str: Information about the counter.
        """
        return str(f"Type: {str(self.counter_type)}\nReps per set: {str(self.reps_per_set)}\nNum sets: {str(self.num_sets)}\nTotal Reps: {str(self.counter)}\nCurrent state: {str(self.state)}")

    def __repr__(self):
        return str(self.output) if self.output else self.get_info()