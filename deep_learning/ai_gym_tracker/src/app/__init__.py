"""Initialize the application."""
import mediapipe as mp
from src.utils import config

# Load config
device = config['video_capture']['device']
flip = config['video_capture']['flip']
show_landmarks = config['video_capture']['show_landmarks']
curl_counter_config = config['pose_estimation']['curl_counter']

# Load MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
