"""Base model class for all counters in the application."""
import traceback
import time
from typing import List, Dict, Optional, Any, Tuple
import cv2
from src.app import mp_pose, text_selected_foreground_image_path, menu_config
from src.app.counters import Counter
from src.utils import numeric_menu_config
from src.utils.utils import SpecialEffects

class ExerciseMenu:
    """
    Class for an interactive exercise menu using hand landmarks.
    """

    SELECTION_HOLD_DURATION = menu_config["selection_hold_duration"]
    HAND_THRESHOLD = menu_config["hand_threshold"]

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

        self.reset()
        self.load_attrs()

    def reset(self) -> None:
        """
        Reset the menu to its initial state.
        """
        self.selected_option = None
        self.selected_reps = None
        self.selected_sets = None
        self.last_selected_num_option = {"selected_reps": None, "selected_sets": None}
        self.right_hand_position = None
        self.left_hand_position = None
        self.start_time = None
        self.landmarks = None
        self.tentative_option = None
        self.state = "start"
        self.state_changed = False
        self.tentative_option_changed = False
        self.output = {}

    def load_attrs(self) -> None:
        """
        Load the attributes for the menu.
        """
        self.images_options = self.prepare_images_options(self.options)
        self.numeric_options = self.prepare_numeric_options(10)

    def calculate_option_position(self, index: int, num_options: int, option_type: str) -> Tuple:
        """
        Calculate the position of an option on the screen.
        """
        section_width = self.width / num_options
        image_width = section_width * 0.5 if option_type == 'image' else section_width

        center_x = (index * section_width) + (section_width / 2)
        center_y = self.height / 2
        return (center_x - (image_width / 2), center_y - (image_width / 2))

    def prepare_images_options(self, options: List[Counter]) -> List[Dict]:
        """
        Prepare the options for the menu, including loading and resizing images.
        """
        prepared_options = []
        num_options = len(options)
        section_width = self.width / num_options

        for i, option in enumerate(options):
            image_width = section_width * 0.5
            prepared_option = self.prepare_single_images_option(option, i, image_width)
            prepared_options.append(prepared_option)
        return prepared_options

    def prepare_single_images_option(self, option: Counter, index: int, image_width: float) -> Dict:
        """
        Prepare a single option, including image loading, resizing, and position calculation.
        """
        image = cv2.imread(option.image_path, cv2.IMREAD_UNCHANGED)
        image_selected = SpecialEffects.draw_selected_halo_from_alpha_channel(
            image=image,
            halo_color=(numeric_menu_config["halo_color"]["b"],
                        numeric_menu_config["halo_color"]["g"],
                        numeric_menu_config["halo_color"]["r"]),
            halo_thickness=numeric_menu_config["halo_thickness"]
        )

        image = cv2.resize(image, (int(image_width), int(image_width)))
        image_selected = cv2.resize(image_selected, (int(image_width), int(image_width)))

        position = self.calculate_option_position(index, len(self.options), 'image')

        return {
            "index": index,
            "image": image,
            "image_selected": image_selected,
            "title": option.title,
            "position": position
        }

    def prepare_numeric_options(self, numeric_options_range: int) -> List[Dict]:
        """
        Prepare numeric options for the menu.
        """
        numeric_options = []
        section_width = self.width / numeric_options_range

        for i in range(numeric_options_range):
            numeric_option = self.prepare_single_numeric_option(i, section_width)
            numeric_options.append(numeric_option)
        return numeric_options

    def prepare_single_numeric_option(self, index: int, section_width: float) -> Dict:
        """
        Prepare a single numeric option, including resizing and position calculation.
        """
        image = cv2.imread(text_selected_foreground_image_path, cv2.IMREAD_UNCHANGED)
        image_selected = SpecialEffects.draw_selected_halo_from_alpha_channel(
            image=image,
            halo_color=(numeric_menu_config["halo_color"]["b"],
                        numeric_menu_config["halo_color"]["g"],
                        numeric_menu_config["halo_color"]["r"]),
            halo_thickness=numeric_menu_config["halo_thickness"]
        )
        image_selected = cv2.resize(image_selected, (int(section_width), int(section_width)))

        position = self.calculate_position_for_number(index, section_width)

        x_offset = 0.3
        y_offset = 0.7
        text_selected_foreground_position = (
            position[0] - int(section_width * x_offset),
            position[1] - int(section_width * y_offset)
        )

        return {
            "index": index,
            "text": str(index + 1),
            "text_selected_foreground": image_selected,
            "position": position,
            "text_selected_foreground_position": text_selected_foreground_position
        }

    def calculate_position_for_number(self, index: int, section_width: float) -> Tuple:
        """
        Calculate screen position for a number option.
        """
        x_offset = section_width / 2 - self.width // 30
        x = int((index * section_width) + x_offset)
        y = int(self.height / 2)
        return (x, y)

    def get_numeric_options_and_positions(self) -> List[Dict[str, Any]]:
        """
        Get the positions of the numeric options.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the position of each numeric option.
        """
        return self.numeric_options

    def get_images_options_and_positions(self) -> List[Dict[str, Any]]:
        """
        Get the images and positions of the exercise options.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the image and position of each exercise option.
        """
        return self.images_options

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
        Update the positions of the hands based on the landmarks detected.
        """
        self.right_hand_position = self.get_hand_position(mp_pose.PoseLandmark.LEFT_INDEX)
        self.left_hand_position = self.get_hand_position(mp_pose.PoseLandmark.RIGHT_INDEX)

    def get_hand_position(self, landmark: mp_pose.PoseLandmark) -> Optional[Tuple]:
        """
        Get the position of a hand based on a given landmark.
        """
        return (
            self.landmarks[landmark.value].x * self.width,
            self.landmarks[landmark.value].y * self.height,
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
            self.check_selection(self.images_options, "selecting_reps", "Exercise")

    def transition_state(self, option: Dict[str, Any], next_state: str) -> None:
        """
        Transition to the next state based on the selected option.
        """
        self.update_selection(option)
        self.state = next_state
        self.reset_selection_state()
        self.state_changed = True
        self.output = self.generate_feedback()

    def update_selection(self, option: Dict[str, Any]) -> None:
        """
        Update the selection based on the current state.
        """
        index = int(option['index'])
        if self.state == "selecting_reps":
            self.selected_reps = index + 1
        elif self.state == "selecting_sets":
            self.selected_sets = index + 1
        else:
            self.selected_option = index

    def reset_selection_state(self) -> None:
        """
        Reset the selection state.
        """
        self.start_time = None
        self.tentative_option = None

    def check_selection(self, options: List[Dict[str, Any]], next_state: str, selection_type: str) -> None:
        """
        Check if hands are over a specific option and handle the selection process.
        """
        hand_over_any_option = False
        current_time = time.time()
        for option in options:
            if self.is_hand_over_option(option):
                hand_over_any_option = True
                if self.tentative_option is None or self.tentative_option != option['index']:
                    self.tentative_option = option['index']
                    self.tentative_option_changed = True
                    self.start_time = current_time
                if current_time - self.start_time >= self.SELECTION_HOLD_DURATION:
                    self.transition_state(option, next_state)
                else:
                    self.update_selection_feedback(option, current_time, selection_type)
                self.last_selected_num_option["selected_reps"] = option['index'] if selection_type == "Reps per set" else None
                self.last_selected_num_option["selected_sets"] = option['index'] if selection_type == "Sets" else None
                break

        if not hand_over_any_option:
            self.reset_selection_state()
            self.generate_selection_feedback(selection_type)

    def update_selection_feedback(self, option: Dict[str, Any], current_time: float, selection_type: str) -> None:
        """
        Update the feedback for the current selection.
        """
        if self.state in ["selecting_reps", "selecting_sets"]:
            self.check_and_reset_start_time(option, current_time, selection_type)
        else:
            time_left = self.SELECTION_HOLD_DURATION - int(current_time - self.start_time)
            self.output = self.generate_feedback(f"Selecting {option['title']} in {time_left} secs")

    def check_and_reset_start_time(self, option: Dict[str, Any], current_time: float, selection_type: str) -> None:
        """
        Check and reset the start time for the selection process.
        """
        option_index = option['index']
        if self.last_selected_num_option["selected_reps"] != option_index and self.last_selected_num_option["selected_sets"] != option_index:
            self.start_time = current_time
        time_left = self.SELECTION_HOLD_DURATION - int(current_time - self.start_time)
        self.output = self.generate_feedback(f"Selecting {selection_type} in {time_left} secs")

    def generate_selection_feedback(self, selection_type: str) -> None:
        """
        Generate feedback based on the selection type.
        """
        if self.state in ["selecting_reps", "selecting_sets"]:
            self.output = self.generate_feedback(f"Select number of {selection_type}")
        else:
            self.output = self.generate_feedback()

    def is_hand_over_option(self, option: Dict[str, Any]) -> bool:
        """
        Determine if the hand is over an option within a certain radius.
        """
        radius = self.calculate_option_radius(option)

        right_hand_distance = self.calculate_hand_distance(self.right_hand_position, option)
        left_hand_distance = self.calculate_hand_distance(self.left_hand_position, option)

        return right_hand_distance <= radius or left_hand_distance <= radius

    def calculate_option_radius(self, option: Dict[str, Any]) -> float:
        """
        Calculate the radius for considering the hand over an option.
        """
        if 'image' in option:
            return self.HAND_THRESHOLD * max(option['image'].shape[0], option['image'].shape[1]) / 2
        text_size = self.calculate_text_size(option)
        return self.HAND_THRESHOLD * text_size[0] * 2

    def calculate_text_size(self, option: Dict[str, Any]) -> Tuple[int, int]:
        """
        Calculate the text size for text options.
        """
        text = option['text']
        font = getattr(cv2, numeric_menu_config["font"])
        font_scale = numeric_menu_config["font_scale"]
        thickness = int(font_scale * 2)
        return cv2.getTextSize(text, font, font_scale, thickness)[0]

    def calculate_hand_distance(self, hand_position: Tuple[float, float], option: Dict[str, Any]) -> float:
        """
        Calculate the distance from the hand to the option's centroid.
        """
        if hand_position is None:
            return float('inf')

        if 'image' in option:
            image = option['image']
            centroid_position = (option['position'][0] + image.shape[1] // 2,
                                 option['position'][1] + image.shape[0] // 2)
        else:
            text_size = self.calculate_text_size(option)
            centroid_position = (option['position'][0] + text_size[0] // 2,
                                 option['position'][1] + text_size[1] // 2)

        return ((hand_position[0] - centroid_position[0]) ** 2 +
                (hand_position[1] - centroid_position[1]) ** 2) ** 0.5

    def set_number_of_reps_and_sets_on_selected_option(self) -> None:
        """
        Set the number of reps and sets on the selected exercise option.
        """
        selected_option = self.options[self.selected_option]
        selected_option.reps_per_set = self.selected_reps
        selected_option.num_sets = self.selected_sets

    def get_selected_option(self) -> Counter:
        """
        Get the currently selected exercise option.
        """
        self.set_number_of_reps_and_sets_on_selected_option()
        return self.options[self.selected_option]

    def run(self, landmarks: List[mp_pose.PoseLandmark]) -> None:
        """
        Run the exercise menu.
        """
        self.landmarks = landmarks
        try:
            self.update_hands_position()
            self.check_hands_position()
        except Exception as e:
            print(e)
            traceback.print_exc()