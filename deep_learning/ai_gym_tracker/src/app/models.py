"""Base model class for all counters in the application."""
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import cv2
from src.app import mp_pose
from src.utils import numeric_menu_config

# Create a Counter base class
class Counter(ABC):
    """
    A base class for Counter objects.

    Attributes
    ------
    title : str
        Title of the counter base class.
    image_path : str
        Path to the exercise example image.
    counter : int
        Current global count.
    state : str
        Current state of the counter.
    output : Dict[str, str]
        Output of the counter.
        Must be in the following format:
        {
            "counter": # Can be a string, int, or list. Lists are used for both sides/arms counters.
            "message": "", # The current message shown to the user.
            "sets": "" # current set / total sets
            "reps": self.reps_per_set, # Only used when counter us list.
        }
    reps_per_set : int
        Number of reps per set.
    num_sets : Optional[int]
        Number of sets.
    current_set : int
        Current set number.
    reps_this_set : int
        Current reps count for the current set.
    landmarks : List[mp_pose.PoseLandmark]
        List of detected landmarks.
    """
    def __init__(
        self,
        title: str,
        image_path: str
    ) -> None:
        """
        Initialize the Counter base class.

        Args:
            title (str): Title of the counter
            image_path (str): Path to the exercise example image
        """
        self.title = title
        self.image_path = image_path

        self.counter: int
        self.state: str
        self.output: Dict[str, str]
        self.reps_per_set: int
        self.num_sets: int
        self.current_set: int
        self.reps_this_set: int
        self.landmarks: List[mp_pose.PoseLandmark]
        self.start_time: Optional[float]
        self.finish_time: Optional[float]

        self.reset_base()

    def reset_base(self):
        """Resets the counter to its initial state."""
        self.counter = 0
        self.state = "start"
        self.output = {}
        self.reps_per_set = 0
        self.num_sets = 0
        self.current_set = 1
        self.reps_this_set = 0
        self.landmarks = []
        self.start_time = None
        self.finish_time = None

    @abstractmethod
    def reset(self):
        """
        Resets the counter to its initial state.
        """
        ...

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

    def increment(self):
        """Increments the counter by 1."""
        self.counter += 1

    def decrement(self):
        """Decrements the counter by 1."""
        self.counter -= 1

    def set_start_time(self) -> None:
        """
        Sets the start time of the counter.
        """
        self.start_time = time.time()

    def get_total_time(self) -> float:
        """
        Returns:
            float: The total time elapsed in seconds since the start of the counter.
        """
        return round(time.time() - self.start_time, 2)
    
    def get_formatted_total_time(self) -> str:
        """
        Returns:
            str: The total time elapsed in minutes and seconds since the start of the counter.
        """
        total_seconds = self.get_total_time()
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes}m {seconds}s"

    def get_total_reps(self) -> int:
        """
        Returns:
            int: The total number of reps.
        """
        if self.state != "finished":
            return self.counter

        return self.reps_per_set * self.num_sets

    def get_info(self):
        """
        Returns:
            str: Information about the counter.
        """
        return str(f"Type: {str(self.title)}\nReps per set: {str(self.reps_per_set)}\nNum sets: {str(self.num_sets)}\nTotal Reps: {str(self.counter)}\nCurrent state: {str(self.state)}")

    def __repr__(self):
        return str(self.output) if self.output else self.get_info()
    
    def is_finished(self) -> bool:
        """
        Returns:
            bool: True if the counter is finished, False otherwise.
        """
        return self.current_set > self.num_sets


class ExerciseMenu:
    """
    Class for an interactive exercise menu using hand landmarks.
    """

    def __init__(self, options: List[Counter], screen_size: Tuple = (1280, 720)) -> None:
        """
        Initialize the ExerciseMenu class.

        Args:
            options (Dict[str, Tuple]): A dictionary mapping exercise names to screen positions (x, y).
        """
        self.options = options
        self.width, self.height = screen_size

        self.selected_option: Optional[int]
        self.selected_reps: Optional[int]
        self.selected_sets: Optional[int]
        self.last_selected_num_option: Dict[str, Optional[int]]
        self.right_hand_position: Optional[Tuple]
        self.left_hand_position: Optional[Tuple]
        self.start_time: Optional[float]
        self.landmarks: Optional[List[mp_pose.PoseLandmark]]
        self.tentative_option: Optional[int]
        self.state: str
        self.state_changed: bool
        self.tentative_option_changed: bool
        self.output: Dict[str, str]

        self.load_attrs()

    def load_attrs(self) -> None:
        """
        Load the attributes for the menu.
        """
        self.reset()
        self.exercise_options = [
            {
                "index" : i,
                "image" : cv2.imread(option.image_path, cv2.IMREAD_UNCHANGED),
                "title" : option.title
            } for i, option in enumerate(self.options.copy())
        ]
        self.calculate_images_positions()
        numeric_options_range = [1, 10]
        self.numeric_options = [
            {
                "index": i,
                "text": str(i),
                "position": self.calculate_position_for_number(i, numeric_options_range[0], numeric_options_range[1]),
            }
            for i in range(numeric_options_range[0], numeric_options_range[1] + 1)
        ]

    def reset(self) -> None:
        """
        Reset the menu to its initial state.
        """
        self.selected_option = None
        self.selected_reps = None
        self.selected_sets = None
        self.last_selected_num_option = {
            "selected_reps": None,
            "selected_sets": None,
        }
        self.right_hand_position = None
        self.left_hand_position = None
        self.start_time = None
        self.landmarks = None
        self.tentative_option = None
        self.state = "start"
        self.state_changed = False
        self.tentative_option_changed = False
        self.output = {}

    def calculate_images_positions(self) -> None:
        """
        Calculate the positions for each exercise image on the screen.
        """
        num_options = len(self.exercise_options)
        section_width = self.width / num_options

        # Set fixed size for exercise images based on section width
        image_width = section_width * 0.5

        # Scale the image height based on the fixed width
        for option in self.exercise_options:
            option["image"] = cv2.resize(option["image"], (int(image_width), int(image_width)))

        for i, option in enumerate(self.exercise_options):
            # Calculate the center x position of the section
            center_x = (i * section_width) + (section_width / 2)
            # Calculate the center y position of the screen
            center_y = self.height / 2

            # Adjust for the size of the image
            image_x = center_x - (image_width / 2)
            image_y = center_y - (image_width / 2)

            # Update the position in the option dictionary
            option["position"] = (image_x, image_y)

    def calculate_position_for_number(self, number: int, start: int, end: int) -> Tuple:
        """
        Calculate screen position for a number option.

        Args:
            number (int): The number for which to calculate the position.
            start (int): Starting number of the range.
            end (int): Ending number of the range.

        Returns:
            Tuple: Position (x, y) on the screen.
        """
        index = number - start
        total_numbers = end - start + 1
        section_width = self.width / total_numbers
        x_offset = section_width / 2 - self.width // 30
        x = int((index * section_width) + x_offset)
        y = int(self.height / 2)
        return (x, y)

    def get_options_images_and_positions(self) -> List[Dict[str, Any]]:
        """
        Get the images and positions of the exercise options.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the image and position of each exercise option.
        """
        return self.exercise_options

    def get_numeric_options_positions(self) -> List[Dict[str, Any]]:
        """
        Get the positions of the numeric options.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the position of each numeric option.
        """
        return self.numeric_options

    def generate_feedback(self, message: str = "Hold palm over an option to select") -> Dict[str, str]:
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
        Check if one of the hands is held over an option (exercise, reps, or sets) and for how long.
        """
        if self.state == "selecting_reps":
            self.check_selection(self.numeric_options, "selecting_sets", "Reps per set")
        elif self.state == "selecting_sets":
            self.check_selection(self.numeric_options, "finished", "Sets")
        else:
            self.check_selection(self.exercise_options, "selecting_reps", "Exercise")

    def get_state_changed(self) -> bool:
        """
        Check if the state has changed.

        Returns:
            bool: True if the state has changed, False otherwise.
        """
        if self.state_changed:
            self.state_changed = False
            return True
        return False
    
    def get_tentative_option_changed(self) -> bool:
        """
        Check if the selected option has changed.

        Returns:
            bool: True if the selected option has changed, False otherwise.
        """
        if self.tentative_option_changed:
            self.tentative_option_changed = False
            return True
        return False

    def check_selection(self, options: List[Dict[str, Any]], next_state: str, selection_type: str) -> None:
        """
        Check if hands are over a specific option and handle the selection process.

        Args:
            options (List[Dict[str, Any]]): List of options to check against.
            next_state (str): The next state to transition to after selection.
            selection_type (str): Type of selection (Exercise, Reps, Sets).
        """
        hand_over_any_option = False
        current_time = time.time()
        for option in options:
            if self.is_hand_over_option(option):
                hand_over_any_option = True
                if self.tentative_option != option['index']:
                    self.tentative_option = option['index']
                    self.tentative_option_changed = True
                    self.start_time = None
                elif self.start_time is None:
                    self.start_time = current_time
                elif current_time - self.start_time >= 3:
                    if self.state == "selecting_reps":
                        self.selected_reps = option['index']
                    elif self.state == "selecting_sets":
                        self.selected_sets = option['index']
                    else:
                        self.selected_option = option['index']
                    self.state = next_state
                    self.start_time = None
                    self.tentative_option = None
                    self.state_changed = True
                    self.output = self.generate_feedback()
                elif self.state in ["selecting_reps", "selecting_sets"]:
                    if self.last_selected_num_option["selected_reps"] != option['index'] and self.last_selected_num_option["selected_sets"] != option['index']:
                        self.start_time = current_time
                    self.output = self.generate_feedback(f"Selecting {option['index']} {selection_type} in {3 - int(current_time - self.start_time)} secs")
                else:
                    self.output = self.generate_feedback(f"Selecting {option['title']} in {3 - int(current_time - self.start_time)} secs")
                self.last_selected_num_option["selected_reps"] = option['index'] if selection_type == "Reps per set" else None
                self.last_selected_num_option["selected_sets"] = option['index'] if selection_type == "Sets" else None
                break

        if not hand_over_any_option:
            self.start_time = None
            self.tentative_option = None
            if self.state in ["selecting_reps", "selecting_sets"]:
                self.output = self.generate_feedback(f"Select number of {selection_type}")
            else:
                self.output = self.generate_feedback()

    def is_hand_over_option(self, option: Dict, threshold: float = 0.9) -> bool:
        """
        Determine if the hand is over an option within a certain radius.

        Args:
            option (dict): The exercise option containing position and image.
            threshold (float): Radius for considering the hand over an option, relative to the image size.

        Returns:
            bool: True if one of the hands is over an option, False otherwise.
        """
        position = option['position']

        if 'image' in option:
            image = option['image']
            # Calculate the centroid of the image
            centroid_position = (position[0] + (image.shape[0] // 2),
                                position[1] + (image.shape[1] // 2))
            
            # Create a circle area around the centroid
            radius = threshold * image.shape[0] / 2
        else:
            # No image, get text size and calculate centroid position
            text = option['text']
            font = getattr(cv2, numeric_menu_config["font"])
            font_scale = numeric_menu_config["font_scale"]
            thickness = font_scale * 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            centroid_position = (position[0] + (text_size[0] // 2),
                                position[1] + (text_size[1] // 2))
            
            # Create a circle area around the centroid with a radius of threshold * text width / 2
            radius = threshold * text_size[0] * 2

        # Calculate the distances of each hand from the centroid
        right_hand_distance = ((self.right_hand_position[0] - centroid_position[0]) ** 2 + (self.right_hand_position[1] - centroid_position[1]) ** 2) ** 0.5
        left_hand_distance = ((self.left_hand_position[0] - centroid_position[0]) ** 2 + (self.left_hand_position[1] - centroid_position[1]) ** 2) ** 0.5

        # Check if either hand is within the radius
        return right_hand_distance <= radius or left_hand_distance <= radius

    def set_number_of_reps_and_sets_on_selected_option(self) -> None:
        """
        Set the number of reps and sets on the selected exercise option.
        """
        setattr(self.options[int(self.selected_option)], "reps_per_set", self.selected_reps)
        setattr(self.options[int(self.selected_option)], "num_sets", self.selected_sets)

    def get_selected_option(self) -> Counter:
        """
        Get the currently selected exercise option.

        Returns:
            str: The selected exercise option, or None if no selection has been made.
        """
        self.set_number_of_reps_and_sets_on_selected_option()
        return self.options[int(self.selected_option)]
    
    def run(self, landmarks: List[mp_pose.PoseLandmark]) -> None:
        """
        Run the exercise menu.

        Args:
            landmarks: The landmarks detected by the pose detection system.
        """
        self.landmarks = landmarks
        self.update_hands_position()
        self.check_hands_position()
