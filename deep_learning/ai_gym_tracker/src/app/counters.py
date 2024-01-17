"""This module contains classes for counting exercises."""
from src.utils.utils import CounterUtils
from src.app.models import Counter
from src.app import mp_pose


# Create curl counter class
class CurlCounter(Counter, CounterUtils):
    """
    Class for counting bicep curls.
    """
    def __init__(
        self,
        min_angle: int,
        max_angle: int,
        start_pose_image_path: str,
        num_sets: int,
        reps_per_set: int,
    ) -> None:
        """
        Class for counting bicep curls.
        
        Args:
            min_angle (int): Minimum angle of the elbow
            max_angle (int): Maximum angle of the elbow
            start_pose_image_path (str): Path to the start pose image
            num_sets (int): Number of sets
            reps_per_set (int): Number of reps per set
        """
        super().__init__(
            reps_per_set=reps_per_set,
            num_sets=num_sets,
            start_pose_image_path=start_pose_image_path)
        self.title = "Bicep Curls"
        self.min_angle = min_angle
        self.max_angle = max_angle

        self.resting_angle = max_angle
        self.current_right_angle = max_angle
        self.current_left_angle = max_angle

        self.right_counter = 0
        self.left_counter = 0

        self.right_state = 'start'
        self.left_state = 'start'

    def get_important_landmarks(self):
        """
        Returns the important landmarks for the exercise.
        """
        return self.landmarks[:25]

    def make_calculations(self):
        """
        Calculates the angle of the elbow and updates the current angle.
        """
        if self.check_starting_pose(self.get_important_landmarks()):
            RIGHT_SHOULDER = [self.landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            RIGHT_ELBOW = [self.landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            RIGHT_WRIST = [self.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            self.current_right_angle = self.calculate_angle_3p(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
            LEFT_SHOULDER = [self.landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, self.landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            LEFT_ELBOW = [self.landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, self.landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            LEFT_WRIST = [self.landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, self.landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            self.current_left_angle = self.calculate_angle_3p(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        else:
            self.state = 'start'
            self.right_state = 'start'
            self.left_state = 'start'
            self.output = {"counter": "", "message": "Please stand in the starting position", "sets": "", "reps": ""}
            self.current_right_angle = self.resting_angle
            self.current_left_angle = self.resting_angle

    def count(self):
        """
        Counts the number of bicep curls.
        """
        if self.state == 'start' and \
            self.right_state == 'start' and \
            self.left_state == 'start' and \
            self.check_starting_pose(self.get_important_landmarks()
        ):
            self.state = 'counting'
            self.right_state = 'down'
            self.left_state = 'down'

        if self.current_right_angle > self.max_angle and self.right_state == 'up':
            self.right_counter += 1
            self.right_state = 'down'
        if self.current_right_angle < self.min_angle and self.right_state == 'down':
            self.right_state = 'up'
            self.output = {
                "counter": [self.left_counter, self.right_counter],
                "sets": {"current": self.current_set, "total": self.num_sets},
                "reps": self.reps_per_set,
                "message": "Slowly curl down right"}
        
        if self.current_left_angle > self.max_angle and self.left_state == 'up':
            self.left_counter += 1
            self.left_state = 'down'
        if self.current_left_angle < self.min_angle and self.left_state == 'down':
            self.left_state = 'up'
            self.output = {
                "counter": [self.left_counter, self.right_counter],
                "sets": {"current": self.current_set, "total": self.num_sets},
                "reps": self.reps_per_set,
                "message": "Slowly curl down left"}

        if self.right_state == 'down' and self.left_state == 'down':
            right_percentage = round(round(1 - (self.current_right_angle - self.min_angle) / (self.max_angle - self.min_angle), 2) * 100)
            left_percentage = round(round(1 - (self.current_left_angle - self.min_angle) / (self.max_angle - self.min_angle), 2) * 100)

            if right_percentage > 0:
                self.output = {"counter": [self.left_counter, self.right_counter],
                "sets": {"current": self.current_set, "total": self.num_sets},
                "reps": self.reps_per_set,
                "message": f"{max(right_percentage, 0)}% right curl up"}
            elif left_percentage > 0:
                self.output = {"counter": [self.left_counter, self.right_counter],
                "sets": {"current": self.current_set, "total": self.num_sets},
                "reps": self.reps_per_set,
                "message": f"{max(left_percentage, 0)}% left curl up"}
            else:
                self.output = {"counter": [self.left_counter, self.right_counter],
                "sets": {"current": self.current_set, "total": self.num_sets},
                "reps": self.reps_per_set,
                "message": "curl up one arm at a time"}

        self.reps_this_set = self.right_counter + self.left_counter
        if self.reps_this_set == self.reps_per_set * 2:
            self.current_set += 1
            self.reps_this_set = 0
            self.right_counter = 0
            self.left_counter = 0
            self.current_right_angle = self.resting_angle
            self.current_left_angle = self.resting_angle
        
        if self.is_finished():
            self.counter = self.reps_per_set * self.num_sets
            self.state = 'finished'
