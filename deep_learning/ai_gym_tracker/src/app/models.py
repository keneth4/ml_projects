"""Base model class for all counters in the application."""
import time
import cv2
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from src.app import mp_pose

# Create a Counter base class
class Counter(ABC):
    """
    A base class for Counter objects.

    Attributes
    ------
    counter : int
        The current global count.
    title : str
        The title of the counter.
    reps_per_set : Optional[int]
        The number of reps per set.
    num_sets : Optional[int]
        The number of sets.
    current_set : int
        The current set.
    reps_this_set : int
        The number of reps this set.
    landmarks : List[mp_pose.PoseLandmark]
        List of detected landmarks.
    state : str
        The current state.
    output : Dict[str, str]
        The output of the counter.
        Must be in the format 
        {
            "counter": # Can be a string, int, or list. Lists are used for both sides/arms counters.
            "message": "", # The current message shown to the user.
            "sets": "" # current set / total sets
            "reps": self.reps_per_set, # Only used when counter us list.
        }
    """
    def __init__(
        self,
        title: str = "",
        image_path: str = "",
        state: str = "start",
        output: Dict[str, str] = None,
        reps_per_set: Optional[int] = None,
        num_sets: Optional[int] = None
    ) -> None:
        """
        Args:
            state (str): The current state.
            output (Dict[str, str]): The output.
            reps_per_set (Optional[int]): The number of reps per set.
            num_sets (Optional[int]): The number of sets.
        """
        self.counter = 0
        self.title = title
        self.image_path = image_path
        self.reps_per_set = reps_per_set
        self.num_sets = num_sets
        self.current_set = 1
        self.reps_this_set = 0

        self.landmarks: List[mp_pose.PoseLandmark] = []
        self.state = state
        self.output = output or {}

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
        return str(f"Type: {str(self.title)}\nReps per set: {str(self.reps_per_set)}\nNum sets: {str(self.num_sets)}\nTotal Reps: {str(self.counter)}\nCurrent state: {str(self.state)}")

    def __repr__(self):
        return str(self.output) if self.output else self.get_info()


class ExerciseMenu:
    """
    Class for an interactive exercise menu using hand landmarks.
    """

    def __init__(self, options: List[Counter], screen_size: tuple = (1280, 720)) -> None:
        """
        Initialize the ExerciseMenu class.

        Args:
            options (Dict[str, tuple]): A dictionary mapping exercise names to screen positions (x, y).
        """
        self.options = options
        self.formated_options = [
            {
                "index" : i,
                "image" : cv2.imread(option.image_path, cv2.IMREAD_UNCHANGED),
                "title" : option.title
            } for i, option in enumerate(options.copy())
        ]
        self.width, self.height = screen_size
        self.selected_option = None
        self.right_hand_position = None
        self.left_hand_position = None
        self.start_time = None
        self.landmarks = None
        self.state = "start"
        self.output = {}
        self.get_images_positions(self.width, self.height)

    def get_images_positions(self, screen_width: int, screen_height: int) -> None:
        """
        Calculate the positions for each exercise image on the screen.

        Args:
            screen_width (int): The width of the screen.
            screen_height (int): The height of the screen.
        """
        num_options = len(self.formated_options)
        section_width = screen_width / num_options

        # Set fixed size for exercise images based on section width
        image_width = section_width * 0.5

        # Scale the image height based on the fixed width
        for option in self.formated_options:
            option["image"] = cv2.resize(option["image"], (int(image_width), int(image_width)))

        for i, option in enumerate(self.formated_options):
            # Calculate the center x position of the section
            center_x = (i * section_width) + (section_width / 2)
            # Calculate the center y position of the screen
            center_y = screen_height / 2

            # Adjust for the size of the image
            image_x = center_x - (image_width / 2)
            image_y = center_y - (image_width / 2)

            # Update the position in the option dictionary
            option["position"] = (image_x, image_y)

    def get_options_images_and_positions(self) -> List[Dict[str, Any]]:
        """
        Get the images and positions of the exercise options.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the image and position of each exercise option.
        """
        return self.formated_options
    
    def generate_feedback(self, message: str = "Hold hand over an option to select") -> Dict[str, str]:
        """
        Generate feedback for the user.
        """
        return {
            "title": "AI Gym Tracker",
            "message": message,
        }

    def update_hands_position(self) -> None:
        """
        Update the positions of the hands.

        Args:
            landmarks: The landmarks detected by the pose detection system.
        """
        self.right_hand_position = (
            self.landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x * self.width,
            self.landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y * self.height,
        )
        self.left_hand_position = (
            self.landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * self.width,
            self.landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * self.height,
        )

    def check_hands_position(self) -> None:
        """
        Check if one of the hands is held over an option and for how long.
        """
        hand_over_any_option = False
        for option in self.formated_options:
            if self.is_hand_over_option(option):
                hand_over_any_option = True
                current_time = time.time()
                if self.start_time is None:
                    self.start_time = current_time
                elif current_time - self.start_time >= 3:  # 3 seconds threshold
                    self.selected_option = self.options[option["index"]]
                    self.state = "finished"
                    return  # Exit the loop and method once an option is selected
                else:
                    self.output = self.generate_feedback(f"Starting {option['title']} in {3 - int(current_time - self.start_time)} secs")

        if not hand_over_any_option:
            # Reset the timer only if neither hand is over any option
            self.state = "start"
            self.start_time = None
            self.output = self.generate_feedback()


    def is_hand_over_option(self, option: dict, threshold: float = 0.9) -> bool:
        """
        Determine if the hand is over an option within a certain radius.

        Args:
            option (dict): The exercise option containing position and image.
            threshold (float): Radius for considering the hand over an option, relative to the image size.

        Returns:
            bool: True if one of the hands is over an option, False otherwise.
        """
        image = option['image']
        position = option['position']
        
        # Calculate the centroid of the image
        centroid_position = (position[0] + (image.shape[0] // 2),
                            position[1] + (image.shape[1] // 2))
        
        # Create a circle area around the centroid with a radius of threshold * image width / 2
        radius = threshold * image.shape[0] / 2

        # Calculate the distances of each hand from the centroid
        right_hand_distance = ((self.right_hand_position[0] - centroid_position[0]) ** 2 + (self.right_hand_position[1] - centroid_position[1]) ** 2) ** 0.5
        left_hand_distance = ((self.left_hand_position[0] - centroid_position[0]) ** 2 + (self.left_hand_position[1] - centroid_position[1]) ** 2) ** 0.5

        # Check if either hand is within the radius
        return right_hand_distance <= radius or left_hand_distance <= radius


    def get_selected_option(self) -> Counter:
        """
        Get the currently selected exercise option.

        Returns:
            str: The selected exercise option, or None if no selection has been made.
        """
        return self.selected_option
    
    def run(self, landmarks: List[mp_pose.PoseLandmark]) -> None:
        """
        Run the exercise menu.

        Args:
            landmarks: The landmarks detected by the pose detection system.
        """
        self.landmarks = landmarks
        self.update_hands_position()
        self.check_hands_position()
