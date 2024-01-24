"""Base model class for all counters in the application."""
import time
from typing import List, Dict, Optional, Any, Tuple
import cv2
from src.app import mp_pose, text_selected_foreground_image_path
from src.app.counters import Counter
from src.utils import numeric_menu_config
from src.utils.utils import VideoCaptureUtils


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
                "image_selected": VideoCaptureUtils.draw_selected_halo_from_alpha_channel(
                    image = cv2.imread(option.image_path, cv2.IMREAD_UNCHANGED),
                    halo_color = (
                        numeric_menu_config["halo_color"]["b"],
                        numeric_menu_config["halo_color"]["g"],
                        numeric_menu_config["halo_color"]["r"]),
                    halo_thickness = numeric_menu_config["halo_thickness"]
                ),
                "title" : option.title
            } for i, option in enumerate(self.options.copy())
        ]
        self.calculate_positions(self.exercise_options, "exercise")
        numeric_options_range = 10
        self.numeric_options = [
            {
                "index": i,
                "text": str(i + 1),
                "text_selected_foreground": VideoCaptureUtils.draw_selected_halo_from_alpha_channel(
                    image = cv2.imread(text_selected_foreground_image_path, cv2.IMREAD_UNCHANGED),
                    halo_color = (
                        numeric_menu_config["halo_color"]["b"],
                        numeric_menu_config["halo_color"]["g"],
                        numeric_menu_config["halo_color"]["r"]),
                    halo_thickness = numeric_menu_config["halo_thickness"]
                ),
            }
            for i in range(numeric_options_range)
        ]
        self.calculate_positions(self.numeric_options, "numeric")

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

    def calculate_images_positions(self, image_index: int, section_width: int, image_width: int) -> Tuple:
        """
        Calculate screen position for an image option.
        """
        # Calculate the center x position of the section
        center_x = (image_index * section_width) + (section_width / 2)
        # Calculate the center y position of the screen
        center_y = self.height / 2

        # Adjust for the size of the image
        image_x = center_x - (image_width / 2)
        image_y = center_y - (image_width / 2)

        return (image_x, image_y)

    def calculate_position_for_number(self, index: int, section_width: int) -> Tuple:
        """
        Calculate screen position for a number option.

        Args:
            number (int): The number for which to calculate the position.
            start (int): Starting number of the range.
            end (int): Ending number of the range.

        Returns:
            Tuple: Position (x, y) on the screen.
        """
        x_offset = section_width / 2 - self.width // 30
        x = int((index * section_width) + x_offset)
        y = int(self.height / 2)
        return (x, y)
    
    def calculate_positions(self, options: List[Dict], option_type: str) -> None:
        """
        Calculate the positions for each option on the screen.

        Args:
            options (list): A list of dictionaries containing the options (either exercise or numeric).
            option_type (str): The type of options ('exercise' or 'numeric').
        """
        num_options = len(options)
        section_width = self.width / num_options

        for i, option in enumerate(options):
            # Calculate the width of the image based on the number of options
            section_width = self.width / num_options
            image_width = section_width * 0.5

            if option_type == 'exercise':
                option["image"] = cv2.resize(option["image"], (int(image_width), int(image_width)))
                option["image_selected"] = cv2.resize(option["image_selected"], (int(image_width), int(image_width)))
                (image_x, image_y) = self.calculate_images_positions(i, section_width, image_width)

            elif option_type == 'numeric':
                # Set fixed size for text based on section width and font size
                (text_x, text_y) = self.calculate_position_for_number(i, section_width)

                # Set fixed size for text foreground based on section width and font size
                option["text_selected_foreground"] = cv2.resize(option["text_selected_foreground"], (int(section_width), int(section_width)))
                x_offset = 0.3
                y_offset = 0.7
                text_selected_foreground_x = text_x - int(section_width * x_offset)
                text_selected_foreground_y = text_y - int(section_width * y_offset)

            else:
                raise ValueError("Unknown option type")

            # Update the position in the option dictionary
            if option_type == 'exercise':
                option["position"] = (image_x, image_y)
            elif option_type == 'numeric':
                option["position"] = (text_x, text_y)
                option["text_selected_foreground_position"] = (text_selected_foreground_x, text_selected_foreground_y)

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
            "tentative_option_index": self.tentative_option,
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

    def transition_state(self, option: Dict[str, Any], next_state: str) -> None:
        """
        Transition to the next state.

        Args:
            option (Dict[str, Any]): The selected option.
            next_state (str): The next state to transition to.
        """
        if self.state == "selecting_reps":
            self.selected_reps = int(option['index']) + 1
        elif self.state == "selecting_sets":
            self.selected_sets = int(option['index']) + 1
        else:
            self.selected_option = option['index']
        self.state = next_state
        self.start_time = None
        self.tentative_option = None
        self.state_changed = True
        self.output = self.generate_feedback()

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
                if self.tentative_option is None or self.tentative_option != option['index']:
                    self.tentative_option = option['index']
                    self.tentative_option_changed = True
                    self.start_time = None
                if self.start_time is None:
                    self.start_time = current_time
                elif current_time - self.start_time >= 3:
                    self.transition_state(option, next_state)
                elif self.state in ["selecting_reps", "selecting_sets"]:
                    if self.last_selected_num_option["selected_reps"] != option['index'] and self.last_selected_num_option["selected_sets"] != option['index']:
                        self.start_time = current_time
                    self.output = self.generate_feedback(f"Selecting {selection_type} in {3 - int(current_time - self.start_time)} secs")
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