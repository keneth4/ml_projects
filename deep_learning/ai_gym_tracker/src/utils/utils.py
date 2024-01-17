"""Utility classes for the AI Gym Tracker application."""
from typing import Tuple, List, Dict
import numpy as np
import cv2

from src.app import mp_pose
from src.app import mp_drawing

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
    
    def check_starting_pose(self, landmarks: List[Dict]) -> bool:
        """
        Checks if the starting pose is correct.

        Args:
            landmarks (List[mp_pose.PoseLandmark]): List of landmarks.

        Returns:
            bool: True if all landmarks in the list have at least 0.9 visibility, False otherwise.
        """
        return all(landmark.visibility > 0.9 for landmark in landmarks)


class VideoCaptureUtils:
    """
    Class with utility functions for capturing video.
    """
    def draw_landmarks(self, image: np.ndarray, landmarks: List[Dict]) -> None:
        """
        Draws the landmarks on an image.

        Args:
            image (np.ndarray): Image to draw on.
            landmarks (List[Dict]): List of landmarks.
        """
        mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


    def get_centered_screen_text_position(self, text: str, font: int, font_scale: float, font_thickness: int, img_size: Tuple[int, int]) -> Tuple[int, int]:
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


    def draw_output_on_image(self, output: Dict[str, str], image: np.ndarray) -> None:
        """
        Draws the output on an image.

        Args:
            output (Dict[str, str]): Output to draw.
            image (np.ndarray): Image to draw on.
        """
        if counter := output.get("counter", ""):
            counter = str(counter)
            # Set parameters for the text
            counter_font = cv2.FONT_HERSHEY_DUPLEX
            counter_font_scale = 5
            counter_font_thickness = counter_font_scale * 2
            counter_border_thickness = counter_font_thickness + 2

            # Calculate text size and position
            counter_text_size = cv2.getTextSize(counter, counter_font, counter_font_scale, counter_font_thickness)[0]
            counter_position = (image.shape[1] - counter_text_size[0] - 50, counter_text_size[1] + 50) # Top-right corner

            # Draw black border by offsetting text
            for x_offset in [-1, 1]:
                for y_offset in [-1, 1]:
                    counter_border_position = (counter_position[0] + x_offset, counter_position[1] + y_offset)
                    cv2.putText(image, counter, counter_border_position, counter_font, counter_font_scale, (0, 0, 0), counter_border_thickness, cv2.LINE_AA)

            # Draw the main text            
            cv2.putText(image, counter, counter_position, counter_font, counter_font_scale, (255, 255, 255), counter_font_thickness, cv2.LINE_AA)

        if message := output.get("message", ""):
            # Set parameters for the text
            message_font = cv2.FONT_HERSHEY_SIMPLEX
            message_font_scale = 2
            message_font_thickness = message_font_scale * 2
            message_border_thickness = message_font_thickness + 2

            # Calculate text size and position
            message_position = self.get_centered_screen_text_position(message, message_font, message_font_scale, message_font_thickness, image.shape)

            # Draw black border by offsetting text
            for x_offset in [-1, 1]:
                for y_offset in [-1, 1]:
                    message_border_position = (message_position[0] + x_offset, message_position[1] + y_offset)
                    cv2.putText(image, message, message_border_position, message_font, message_font_scale, (0, 0, 0), message_border_thickness, cv2.LINE_AA)

            # Draw the main text
            cv2.putText(image, message, message_position, message_font, message_font_scale, (255, 255, 255), message_font_thickness, cv2.LINE_AA)


    def draw_start_pose(self, start_pose_image: np.ndarray, image: np.ndarray, opacity: float) -> np.ndarray:
        """
        Draws the start pose on an image with a given opacity.

        Args:
            pose_img_path (str): Path to the start pose image.
            image (np.ndarray): Image to draw on.
        """
        # Check if the image has an alpha channel
        if start_pose_image.shape[2] == 4:
            # Extract the alpha channel as a mask
            alpha_channel = start_pose_image[:,:,3]

        # Convert to BGR
        start_pose_image = cv2.cvtColor(start_pose_image, cv2.COLOR_BGRA2BGR)

        # Calculate the ratio of the new height to the old height
        ratio = image.shape[0] / start_pose_image.shape[0]

        # Calculate new width and height
        new_height = image.shape[0]
        new_width = int(start_pose_image.shape[1] * ratio)

        # Resize the image, preserving the aspect ratio
        start_pose_image = cv2.resize(start_pose_image, (new_width, new_height))

        # Resize the alpha channel to match the resized image
        alpha_channel_resized = cv2.resize(alpha_channel, (new_width, new_height))

        # Since the resized image might not match the background size, we need to center it
        # Create a new image with the same size as the background and fill it with white color
        silhouette = np.ones_like(image) * 255

        # Calculate top-left corner position to center the silhouette on the background
        x_offset = (image.shape[1] - new_width) // 2
        y_offset = (image.shape[0] - new_height) // 2

        # Create mask for the area we want to blend on the background based on the new size of the start pose image
        mask = alpha_channel_resized > 0

        # Place the resized start pose image onto the silhouette image at the calculated offset
        silhouette[y_offset:y_offset+new_height, x_offset:x_offset+new_width][mask] = start_pose_image[mask]

        # Now blend the silhouette with the background image
        return cv2.addWeighted(silhouette, opacity, image, 1 - opacity, 0)
