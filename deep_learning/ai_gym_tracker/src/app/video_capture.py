"""Video capture class."""
import contextlib
import cv2

from src.utils.utils import VideoCaptureUtils
from src.app.models import Counter
from src.app import mp_pose, mp_drawing


# Create a video capture class, generalizing the code above
class PoseDetectorVideoCapture(VideoCaptureUtils):
    """
    Class for capturing video from a camera, detecting poses, and displaying the results.

    Attributes
    ----------
    cap : cv2.VideoCapture
        The video capture object.
    screen_width : int
        The width of the screen.
    screen_height : int
        The height of the screen.
    flip : bool
        Whether to flip the video feed horizontally.
    """
    def __init__(self, device: int = 0, flip: bool = False, show_landmarks: bool = True) -> None:
        """
        Args:
            device (int): The device index of the camera to use.
        """
        self.cap = cv2.VideoCapture(device)
        self.screen_width = int(self.cap.get(3))
        self.screen_height = int(self.cap.get(4))
        self.flip = flip
        self.show_landmarks = show_landmarks

        from src.utils import config
        # Load config
        self.MIN_DETECTION_CONFIDENCE = config['mediapipe']['pose']['min_detection_confidence']
        self.MIN_TRACKING_CONFIDENCE = config['mediapipe']['pose']['min_tracking_confidence']

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
        with mp_pose.Pose(min_detection_confidence=self.MIN_DETECTION_CONFIDENCE, min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE) as pose:
            while self.cap.isOpened():
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
                    landmarks = results.pose_landmarks.landmark
                    
                    # Run pose counter
                    pose_counter.run(landmarks)

                    # Render pose counter
                    self.draw_counter_on_image(pose_counter.output["counter"], image)
                    self.draw_text_on_image(pose_counter.output["message"], image)

                # Render detections
                if self.show_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                    )

                # Show detections
                cv2.imshow('Detection Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
