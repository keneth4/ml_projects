"""Main application file."""
import traceback

from src.app.video_capture import PoseDetectorVideoCapture
from src.app.counters import CurlCounter, SquatCounter

from src.app import device, flip, show_landmarks
from src.app import curl_counter_config, squat_counter_config


def main() -> None:
    """Main application function."""

    # Start PoseDetectorVideoCapture
    with PoseDetectorVideoCapture(device=device, flip=flip, show_landmarks=show_landmarks) as video_capture:
        # Load exercise counters
        video_capture.load_options([
            CurlCounter(**curl_counter_config),
            SquatCounter(**squat_counter_config)
        ])

        # Run menu
        while chosen_counter := video_capture.run_menu():
            # Load counter
            video_capture.load_counter(chosen_counter)

            # Run video capture
            video_capture.run_counter()

            # Print total reps
            print(f"Total {chosen_counter.title} reps: {chosen_counter.get_total_reps()}")

            # Reset counter and menu
            chosen_counter.reset()
            video_capture.menu.reset()

            if video_capture.exit:
                break

if __name__ == "__main__":
    try:
        # Run main application
        main()
    except Exception as e:
        # Print exception
        print(e)

        # Print traceback
        traceback.print_exc()

        # Exit
        exit()
