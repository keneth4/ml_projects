"""Video capture class."""
import contextlib
import cv2

from src.utils.utils import VideoCaptureUtils
from src.app.models import Counter
from src.app import mp_pose, min_detection_confidence, min_tracking_confidence, window_name


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
        self.flip = flip
        self.show_landmarks = show_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()
        cv2.destroyAllWindows()
        
    def run(self, pose_counter: Counter) -> None:
        """
        Runs the pose detector video capture.

        Args:
            pose_counter (Counter): The pose counter to use.
        """
        with mp_pose.Pose(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as pose:
            start_pose_image = cv2.imread(pose_counter.start_pose_image_path, cv2.IMREAD_UNCHANGED)
            while self.cap.isOpened() and pose_counter.state != "finished":
                _, image = self.cap.read()

                # Flip image horizontally
                if self.flip:
                    image = cv2.flip(image, 1)

                # Recolor feed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor image back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                with contextlib.suppress(Exception):
                    landmarks = results.pose_landmarks
                    
                    # Run pose counter
                    pose_counter.run(landmarks.landmark)

                    # Render start pose
                    if pose_counter.state == "start":
                        image = self.draw_start_pose(start_pose_image, image, opacity=0.3)
                    else:
                        self.draw_stats_background(image)
                    
                    # Render output
                    self.draw_output_on_image(pose_counter.title, pose_counter.output, image)

                    # Render detections
                    if self.show_landmarks:
                        self.draw_landmarks(image, landmarks)

                # Show to screen
                cv2.imshow(window_name, image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break