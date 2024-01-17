"""Initialize the application."""
import mediapipe as mp
from src.utils import config

# Load VideoCapture config
device = config['video_capture']['device']
flip = config['video_capture']['flip']
min_detection_confidence = config['mediapipe']['pose']['min_detection_confidence']
min_tracking_confidence = config['mediapipe']['pose']['min_tracking_confidence']
show_landmarks = config['video_capture']['show_landmarks']

# Load CurlCounter config
curl_counter_config = config['pose_estimation']['curl_counter']

# Load MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
