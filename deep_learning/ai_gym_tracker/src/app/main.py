"""Main application file."""
from src.app.video_capture import PoseDetectorVideoCapture
from src.app.counters import CurlCounter

from src.app import DEVICE, FLIP, SHOW_LANDMARKS

if __name__ == "__main__":
    # Start PoseDetectorVideoCapture
    with PoseDetectorVideoCapture(device=DEVICE, flip=FLIP, show_landmarks=SHOW_LANDMARKS) as video_capture:
        # Create instance of CurlCounter
        curl_counter = CurlCounter()

        # Run video capture
        video_capture.run(curl_counter)