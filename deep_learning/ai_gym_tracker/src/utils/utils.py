"""Utility classes for the AI Gym Tracker application."""
from typing import Tuple
import numpy as np
import cv2

from src.app.models import Counter
from src.app import mp_pose

# Create counters utils class
class CounterUtils:
    """
    Class with utility functions for counting reps and sets in a workout.
    """
    def calculate_angle_3p(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> int:
        """
        Calculates the angle between three given points.

        Args:
            a (Tuple[float, float]): First point
            b (Tuple[float, float]): Second point
            c (Tuple[float, float]): Third point

        Returns:
            int: Angle between the three given points
        """
        # Calculating the vectors ab and bc
        ab = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])

        # Calculating the cosine of the angle using the dot product
        cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        
        # Ensuring the cosine value is within the valid range [-1, 1]
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        # Calculating the angle in radians and converting it to degrees
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def is_visible(self, landmark: mp_pose.PoseLandmark) -> bool:
        """
        Checks if a landmark is visible.
        """
        return self.landmarks[landmark.value].visibility > 0.5


class VideoCaptureUtils:
    """
    Class with utility functions for capturing video.
    """
    def get_screen_text_position(self, text: str, font: int, font_scale: float, font_thickness: int, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculates the position of the text on the screen.

        Args:
            text (str): Text to draw on the image.
            font (int): Font to use.
            font_scale (float): Font scale to use.
            font_thickness (int): Font thickness to use.
            img_size (Tuple[int, int]): Size of the image.

        Returns:
            Tuple[int, int]: Position of the text on the screen.
        """
        # Calculate text size and position
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (img_size[0]) - (text_size[0] // 2) - 50 # 50 is a magic number to center the text
        text_y = (img_size[1] // 2) + (text_size[1] // 2)
        return (text_x, text_y)

    def draw_text_on_image(self, text: str, image: np.ndarray, bottom: bool = True, font_scale: float = 1) -> None:
        """
        Draws text with a black border on an image.

        Args:
            text (str): Text to draw on the image.
            image (np.ndarray): Image to draw on.
        """

        # Set parameters for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = font_scale
        font_thickness = font_scale * 2
        border_thickness = font_thickness + 2

        # Calculate text size and position
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        position = self.get_screen_text_position(text, font, font_scale, font_thickness, image.shape)

        if not bottom:
            position = (position[0], text_size[1] + 50)

        # Draw black border by offsetting text
        for x_offset in [-1, 1]:
            for y_offset in [-1, 1]:
                border_position = (position[0] + x_offset, position[1] + y_offset)
                cv2.putText(image, text, border_position, font, font_scale, (0, 0, 0), border_thickness, cv2.LINE_AA)

        # Draw the main text
        cv2.putText(image, text, position, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    def draw_counter_on_image(self, counter: Counter, image: np.ndarray) -> None:
        """
        Draws the counter on an image.

        Args:
            counter (Counter): Counter to draw.
            image (np.ndarray): Image to draw on.
        """
        self.draw_text_on_image(str(counter), image, bottom=False, font_scale=3)
