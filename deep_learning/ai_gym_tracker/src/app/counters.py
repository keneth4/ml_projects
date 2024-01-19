"""This module contains classes for counting exercises."""
import numpy as np
from typing import Dict
from src.utils.utils import CounterUtils
from src.app.models import Counter
from src.app import mp_pose


class CurlCounter(Counter, CounterUtils):
    """
    Class for counting bicep curls.
    """
    def __init__(
        self,
        min_angle: int,
        max_angle: int,
        num_sets: int,
        reps_per_set: int,
    ) -> None:
        """
        Class for counting bicep curls.
        
        Args:
            min_angle (int): Minimum angle of the elbow
            max_angle (int): Maximum angle of the elbow
            num_sets (int): Number of sets
            reps_per_set (int): Number of reps per set
        """
        super().__init__(
            title="Bicep Curls",
            reps_per_set=reps_per_set,
            num_sets=num_sets)
        self.min_angle = min_angle
        self.max_angle = max_angle

        self.current_right_angle = max_angle
        self.current_left_angle = max_angle

        self.right_counter = 0
        self.left_counter = 0

        self.right_state = 'start'
        self.left_state = 'start'


    def get_used_landmarks(self):
        """
        Returns the important landmarks for the exercise.
        """
        return self.landmarks[:25] # From nose to right hip
    

    def generate_feedback(self, message: str, display_counters: bool = True) -> Dict[str, str]:
        """
        Formats the feedback to be displayed to the user.
        """
        return {
            "counter": [self.left_counter, self.right_counter] if display_counters else [0,0],
            "sets": f"Set {self.current_set}/{self.num_sets}",
            "reps": self.reps_per_set,
            "message": message
        }
    

    def set_starting_pose(self, message: str = "Please stand in the starting position") -> None:
        """
        Resets the state of the counter.
        """
        self.state = 'start'
        self.right_state = 'start'
        self.left_state = 'start'
        self.output = self.generate_feedback(message=message)
        self.current_right_angle = self.max_angle
        self.current_left_angle = self.max_angle


    def arm_counting_cycle(self, side: str) -> None:
        """
        Counts the number of bicep curls for a single arm.
        """
        if self.__getattribute__(f"current_{side}_angle") > self.max_angle and self.__getattribute__(f"{side}_state") == 'up':
            self.__setattr__(f"{side}_counter", self.__getattribute__(f"{side}_counter") + 1)
            self.__setattr__(f"{side}_state", 'down')
        if self.__getattribute__(f"current_{side}_angle") < self.min_angle and self.__getattribute__(f"{side}_state") == 'down':
            self.__setattr__(f"{side}_state", 'up')
            self.output = self.generate_feedback("Slowly curl down")


    def check_completed_set(self) -> None:
        """
        Update and reset counters after a set is completed.
        """
        self.reps_this_set = self.right_counter + self.left_counter
        if self.reps_this_set == self.reps_per_set * 2:
            self.current_set += 1
            self.reps_this_set = 0
            self.right_counter = 0
            self.left_counter = 0
            self.current_right_angle = self.max_angle
            self.current_left_angle = self.max_angle


    def make_calculations(self) -> None:
        """
        Calculates the angle of the elbow and updates the current angle.
        """
        # Make sure the user is in the starting position
        if not self.check_starting_pose(self.get_used_landmarks()):
            self.set_starting_pose()
            return
        
        # Calculate the angle of the right elbow
        RIGHT_SHOULDER = [self.landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        RIGHT_ELBOW = [self.landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        RIGHT_WRIST = [self.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        self.current_right_angle = self.calculate_angle_3p(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)

        # Calculate the angle of the left elbow
        LEFT_SHOULDER = [self.landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, self.landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        LEFT_ELBOW = [self.landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, self.landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        LEFT_WRIST = [self.landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, self.landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        self.current_left_angle = self.calculate_angle_3p(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)


    def count(self) -> None:
        """
        Counts the number of bicep curls.
        """
        # Starting position -> counting position
        if self.state == 'start' and self.check_starting_pose(self.get_used_landmarks()):
            self.state = 'counting'
            self.right_state = 'down'
            self.left_state = 'down'

        # Both arms counting cycles
        self.arm_counting_cycle('right')
        self.arm_counting_cycle('left')

        # While both arms are down, display the percentage of the curl up
        if self.right_state == 'down' and self.left_state == 'down':
            right_percentage = round(round(1 - (self.current_right_angle - self.min_angle) / (self.max_angle - self.min_angle), 2) * 100)
            left_percentage = round(round(1 - (self.current_left_angle - self.min_angle) / (self.max_angle - self.min_angle), 2) * 100)

            # Check if both arms are up at the same time
            if right_percentage > 0 and left_percentage > 0:
                self.set_starting_pose(message="Please curl up one arm at a time!")

            # Check if both arms are down at the same time
            elif right_percentage <= 0 and left_percentage <= 0:
                self.output = self.generate_feedback("curl up one arm at a time")

            # Display the percentage of the curl up if only one arm is up
            elif right_percentage > 0 or left_percentage > 0:
                percentage = max(right_percentage if right_percentage > 0 else 0, left_percentage if left_percentage > 0 else 0)
                self.output = self.generate_feedback(f"{percentage}% curl up")

        # Check if the user is finished with the current set
        self.check_completed_set()

        # Change state to finished and set counter to max reps if the user is finished
        if self.is_finished():
            self.counter = self.reps_per_set * self.num_sets
            self.state = 'finished'


class SquatCounter(Counter, CounterUtils):
    """
    Class for counting squats.
    """
    def __init__(
        self,
        min_angle: int,
        max_angle: int,
        num_sets: int,
        reps_per_set: int,
    ) -> None:
        """
        Class for counting squats.

        Args:
            min_angle (int): Minimum angle of the knees (usually at the bottom of the squat)
            max_angle (int): Maximum angle of the knees (standing position)
            num_sets (int): Number of sets
            reps_per_set (int): Number of reps per set
        """
        super().__init__(
            title="Squats",
            reps_per_set=reps_per_set,
            num_sets=num_sets)
        self.min_angle = min_angle
        self.max_angle = max_angle

        self.current_angle = max_angle
        self.counter = 0
        self.state = 'start'


    def get_used_landmarks(self):
        """
        Returns the important landmarks for the squat exercise.
        """
        return self.landmarks[23:29] # From left hip to right ankle
    

    def generate_feedback(self, message: str, display_counter: bool = True) -> Dict[str, str]:
        """
        Formats the feedback to be displayed to the user.
        """
        return {
            "counter": f"{self.reps_this_set}/{self.reps_per_set}" if display_counter else 0,
            "sets": f"Set {self.current_set}/{self.num_sets}",
            "message": message
        }
    

    def set_starting_pose(self, message: str = "Please stand in the starting position") -> None:
        """
        Resets the state of the counter.
        """
        self.state = 'start'
        self.output = self.generate_feedback(message=message)
        self.current_angle = self.max_angle


    def squat_counting_cycle(self) -> None:
        """
        Counts the number of squats.
        """
        if self.current_angle >= self.max_angle and self.state == 'down':
            self.state = 'up'
            self.counter += 1
        elif self.current_angle <= self.min_angle and self.state == 'up':
            self.state = 'down'


    def check_completed_set(self) -> None:
        """
        Update and reset counters after a set is completed.
        """
        self.reps_this_set = self.counter - (self.reps_per_set * (self.current_set - 1))
        if self.reps_this_set == self.reps_per_set:
            self.current_set += 1
            self.reps_this_set = 0
            self.reps_this_set = 0
            self.current_angle = self.max_angle


    def make_calculations(self) -> None:
        """
        Calculates the angle of the knees and updates the current angle.
        """
        # Make sure the user is in the starting position
        if not self.check_starting_pose(self.get_used_landmarks()):
            self.set_starting_pose()
            return

        # Calculate the angle of the left knee
        LEFT_HIP = [self.landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, self.landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        LEFT_KNEE = [self.landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, self.landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        LEFT_ANKLE = [self.landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, self.landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        current_left_knee_angle = self.calculate_angle_3p(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)

        # Calculate the angle of the right knee
        RIGHT_HIP = [self.landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        RIGHT_KNEE = [self.landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        RIGHT_ANKLE = [self.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        current_right_knee_angle = self.calculate_angle_3p(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)

        # Update the current angle to the average of the left and right knee angles
        self.current_angle = np.mean([current_left_knee_angle, current_right_knee_angle])


    def count(self) -> None:
        """
        Counts the number of squats.
        """
        # Starting position -> counting position
        if self.state == 'start' and self.check_starting_pose(self.get_used_landmarks()):
            self.state = 'up'

        # Squat counting cycle
        self.squat_counting_cycle()

        # Display the percentage of the squat down
        if self.state == 'up':
            percentage = round(round(1 - (self.current_angle - self.min_angle) / (self.max_angle - self.min_angle), 2) * 100)
            self.output = self.generate_feedback(f"{max(percentage, 0)}% squat down")

        # Display the percentage of the squat up
        elif self.state == 'down':
            percentage = round(round(1 - (self.max_angle - self.current_angle) / (self.max_angle - self.min_angle), 2) * 100)
            if percentage > 0:
                self.output = self.generate_feedback(f"{max(percentage, 0)}% squat up")
            else:
                self.output = self.generate_feedback("Get ready to stand up")

        # Check if the user is finished with the current set
        self.check_completed_set()

        # Change state to finished and set counter to max reps if the user is finished
        if self.is_finished():
            self.state = 'finished'
