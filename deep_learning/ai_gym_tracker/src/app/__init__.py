"""Initialize the application."""
import mediapipe as mp
from src.utils import config

# Load config
DEVICE = config['video_capture']['device']
FLIP = config['video_capture']['flip']
SHOW_LANDMARKS = config['video_capture']['show_landmarks']

# Load MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
