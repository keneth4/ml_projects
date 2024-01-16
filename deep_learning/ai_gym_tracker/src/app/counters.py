"""This module contains classes for counting exercises."""
from src.utils.utils import CounterUtils
from src.app.models import Counter
from src.app import mp_pose


# Create curl counter class
class CurlCounter(Counter, CounterUtils):
    """
    Class for counting bicep curls.
    """
    def __init__(self, min_angle=30, max_angle=150):
        """
        Class for counting bicep curls.
        
        Args:
            min_angle (int): Minimum angle of the elbow
            max_angle (int): Maximum angle of the elbow
            resting_angle (int): Resting angle of the elbow

        Attributes:
            min_angle (int): Minimum angle of the elbow
            max_angle (int): Maximum angle of the elbow
            resting_angle (int): Resting angle of the elbow
            current_angle (int): Current angle of the elbow
            LEFT_SHOULDER (list): Coordinates of the left shoulder
            LEFT_ELBOW (list): Coordinates of the left elbow
            LEFT_WRIST (list): Coordinates of the left wrist
        """
        super().__init__(state='down')
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.resting_angle = max_angle
        self.current_angle = max_angle
        self.LEFT_SHOULDER = None
        self.LEFT_ELBOW = None
        self.LEFT_WRIST = None

    def make_calculations(self):
        """
        Calculates the angle of the elbow and updates the current angle.
        """
        if self.is_visible(mp_pose.PoseLandmark.LEFT_SHOULDER) \
        and self.is_visible(mp_pose.PoseLandmark.LEFT_ELBOW) \
        and self.is_visible(mp_pose.PoseLandmark.LEFT_WRIST):
            self.LEFT_SHOULDER = [self.landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            self.LEFT_ELBOW = [self.landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            self.LEFT_WRIST = [self.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, self.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            self.current_angle = self.calculate_angle_3p(self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST)
        else:
            self.current_angle = self.resting_angle

    def count(self):
        """
        Counts the number of bicep curls.
        """
        if self.current_angle > self.max_angle and self.state == 'up':
            self.increment()
            self.state = 'down'
        if self.current_angle < self.min_angle and self.state == 'down':
            self.state = 'up'
            self.output = {"counter": self.counter, "message": "Slowly return to starting position"}
        if self.state == 'down':
            percentage = round(round(1 - (self.current_angle - self.min_angle) / (self.max_angle - self.min_angle), 2) * 100)
            self.output = {"counter": self.counter, "message": f"{max(percentage, 0)}% to complete next rep"}
