"""Initialize the application."""
import mediapipe as mp
from src.utils import config

# Load VideoCapture config
device = config['video_capture']['device']
flip = config['video_capture']['flip']
min_detection_confidence = config['mediapipe']['pose']['min_detection_confidence']
min_tracking_confidence = config['mediapipe']['pose']['min_tracking_confidence']
show_landmarks = config['video_capture']['show_landmarks']

# Load interface config
window_name = config['interface']['window_name']
stats_background_color = (
    config['interface']['stats_bar']['background_color']['r'],
    config['interface']['stats_bar']['background_color']['g'],
    config['interface']['stats_bar']['background_color']['b'])
stats_position_top = config['interface']['stats_bar']['position_top']
start_pose_image_path = config['interface']['start_pose_image_path']

# Load counters config
curl_counter_config = config['pose_estimation']['curl_counter']
squat_counter_config = config['pose_estimation']['squat_counter']

# Load MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
