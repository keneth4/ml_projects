"""Video capture class."""
import contextlib
from typing import List, Tuple
import numpy as np
import cv2

from src.utils.utils import VideoCaptureUtils
from src.app.models import Counter, ExerciseMenu
from src.app import mp_pose, min_detection_confidence, min_tracking_confidence, window_name, start_pose_image_path


# Create a video capture class, generalizing the code above
class PoseDetectorVideoCapture(VideoCaptureUtils):
    """
    Class for capturing video from a camera, detecting poses, and displaying the results.

    Attributes
    ----------
    cap : cv2.VideoCapture
        The video capture object.
    flip : bool
        Whether to flip the video feed horizontally.
    show_landmarks : bool
        Whether to show the landmarks on the video feed.
    min_detection_confidence : float
        Minimum confidence value ([0.0, 1.0]) for pose detection to be considered successful.
    min_tracking_confidence : float
        Minimum confidence value ([0.0, 1.0]) for pose tracking to be considered successful.
    """
    def __init__(self, device: int = 0, flip: bool = False, show_landmarks: bool = True) -> None:
        """
        Args:
            device (int): The device index of the camera to use.
        """
        self.cap = cv2.VideoCapture(device)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.flip = flip
        self.show_landmarks = show_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.options: List[Counter] = []
        self.menu: ExerciseMenu = None
        self.pose = mp_pose.Pose(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence)
        self.counter: Counter = None
        self.start_pose_image: np.ndarray = cv2.imread(start_pose_image_path, cv2.IMREAD_UNCHANGED)
        self.exit = False

        print(f"VideoCapture: {self.width}x{self.height}")


    def __enter__(self):
        self.pose.__enter__()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()
        cv2.destroyAllWindows()


    def load_options(self, options: List[Counter]) -> None:
        """
        Loads the options to choose from.

        Args:
            options (List[Counter]): The options to choose from.
        """
        self.options = options
        print(f"Loaded {[option.title for option in self.options]}")
        self.menu = ExerciseMenu(options, (self.width, self.height))


    def process_result_frame(self, image: np.ndarray) -> Tuple[np.ndarray, mp_pose.PoseLandmark]:
        """
        Processes a frame from the video capture.

        Args:
            image (np.ndarray): The image to process.

        Returns:
            mp_pose.PoseLandmark: The landmark of the detected pose.
        """
        # Flip image horizontally
        if self.flip:
            image = cv2.flip(image, 1)

        # Recolor feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = self.pose.process(image)

        # Recolor image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return (image, results)
    

    def process_menu_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Processes a frame from the video capture.

        Args:
            image (np.ndarray): The image to process.

        Returns:
            Union[Counter, np.ndarray]: The chosen counter or the processed image.
        """
        image, results = self.process_result_frame(image)

        # Extract landmarks
        with contextlib.suppress(Exception):
            landmarks = results.pose_landmarks

            # Run menu
            self.menu.run(landmarks.landmark)

            # Render menu
            if self.menu.state == "start":
                image = self.draw_menu_images(self.menu.get_options_images_and_positions(), image)
            else:
                self.draw_numeric_menu(self.menu.get_numeric_options_positions(), image)

            # Render output
            self.draw_output_on_image(self.menu.output, image)

            # Render detections
            if self.show_landmarks:
                self.draw_landmarks(image, landmarks)

        if self.menu.state == "finished":
            self.counter = self.menu.get_selected_option()

        return image
    
    def run_menu(self) -> Counter:
        """
        Runs the pose detector video capture menu.

        Args:
            options (List[Counter]): The options to choose from.

        Returns:
            Counter: The chosen option.
        """
        while self.cap.isOpened():
            _, frame = self.cap.read()

            # Process frame
            frame = self.process_menu_frame(frame)
            
            # check if counter is chosen
            if self.menu.state == "finished":
                return self.counter

            # Show to screen
            cv2.imshow(window_name, frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


    def process_counter_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Processes a frame from the video capture.

        Args:
            image (np.ndarray): The image to process.

        Returns:
            Union[Counter, np.ndarray]: The chosen counter or the processed image.
        """
        image, results = self.process_result_frame(image)

        # Extract landmarks
        with contextlib.suppress(Exception):
            landmarks = results.pose_landmarks

            # Run pose counter
            self.counter.run(landmarks.landmark)

            # Render start pose
            if self.counter.state == "start":
                image = self.draw_start_pose(self.start_pose_image, image, opacity=0.5)
            else:
                self.draw_stats_background(image)
            
            # Render output
            self.draw_output_on_image(self.counter.output, image)

            # Render detections
            if self.show_landmarks:
                self.draw_landmarks(image, landmarks)

        return image


    def load_counter(self, counter: Counter) -> None:
        """
        Loads the counter to use.

        Args:
            counter (Counter): The counter to use.
        """
        self.counter = counter


    def run_counter(self) -> None:
        """
        Runs the pose detector video capture.
        """
        while self.cap.isOpened() and self.counter.state != "finished":
            _, frame = self.cap.read()

            # Process frame
            frame = self.process_counter_frame(frame)

            # Show to screen
            cv2.imshow(window_name, frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                self.exit = True
                break
