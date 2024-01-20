"""Main application file."""
from src.app.video_capture import PoseDetectorVideoCapture
from src.app.counters import CurlCounter, SquatCounter

from src.app import device, flip, show_landmarks
from src.app import curl_counter_config, squat_counter_config

if __name__ == "__main__":
    # Start PoseDetectorVideoCapture
    with PoseDetectorVideoCapture(device=device, flip=flip, show_landmarks=show_landmarks) as video_capture:
        # Load exercise counters
        exercise_counters = [
            CurlCounter(**curl_counter_config),
            SquatCounter(**squat_counter_config)
        ]

        # Run menu
        while counter := video_capture.run_menu(exercise_counters):
            # Run video capture
            video_capture.run_counter(counter)
            total_reps = counter.get_total_reps()
            counter.reset()
            print(f"Total {counter.title} reps: {total_reps}")