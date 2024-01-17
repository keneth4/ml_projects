"""Main application file."""
from src.app.video_capture import PoseDetectorVideoCapture
from src.app.counters import CurlCounter

from src.app import device, flip, show_landmarks
from src.app import curl_counter_config

if __name__ == "__main__":
    # Start PoseDetectorVideoCapture
    with PoseDetectorVideoCapture(window_name='AI Gym Tracker', device=device, flip=flip, show_landmarks=show_landmarks) as video_capture:
        # Create instance of CurlCounter
        curl_counter = CurlCounter(**curl_counter_config, num_sets=2, reps_per_set=5)

        # Run video capture
        video_capture.run(curl_counter)