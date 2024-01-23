"""Initialize the application."""
import mediapipe as mp
from src.utils import config

# Load VideoCapture config
video_capture_config = config['video_capture']
device = video_capture_config['device']
flip = video_capture_config['flip']
show_landmarks = video_capture_config['show_landmarks']

# Load MediaPipe Pose config
pose_config = config['mediapipe']['pose']
min_detection_confidence = pose_config['min_detection_confidence']
min_tracking_confidence = pose_config['min_tracking_confidence']

# Load interface config
interface_config = config['interface']
window_name = interface_config['window_name']
start_pose_image_path = interface_config['start_pose_image_path']

# Load sound config
sound_config = interface_config['sound']

# Load stats bar config
stats_bar_config = interface_config['stats_bar']
stats_background_color = (
    stats_bar_config['background_color']['r'],
    stats_bar_config['background_color']['g'],
    stats_bar_config['background_color']['b'])

# Load counters config
pose_estimation_config = config['pose_estimation']
curl_counter_config = pose_estimation_config['curl_counter']
squat_counter_config = pose_estimation_config['squat_counter']

# Load MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
