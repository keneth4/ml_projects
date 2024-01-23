"""Utility classes for the AI Gym Tracker application."""
from typing import Tuple, List, Dict, Union
import numpy as np
import simpleaudio as sa
import cv2

from src.app import mp_pose, mp_drawing, stats_background_color
from src.utils import (
    counter_config,
    double_counter_config,
    message_config,
    title_config,
    sets_config,
    numeric_menu_config,
    timer_config,
    stats_config)


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
    def get_bottom_center_screen_text_position(self, text: str, font: int, font_scale: float, font_thickness: int, img_size: Tuple[int, int]) -> Tuple[int, int]:
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
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        return ((img_size[0]) - (text_size[0] // 2) - 75, (img_size[1] // 2) + (text_size[1] // 2))

    def get_top_center_screen_text_position(self, text: str, font: int, font_scale: float, font_thickness: int, img_size: Tuple[int, int]) -> Tuple[int, int]:
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
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        return ((img_size[0]) - (text_size[0] // 2) - 75, text_size[1] + 40)

    def get_top_right_screen_text_position(self, text: str, font: int, font_scale: float, font_thickness: int, img_size: Tuple[int, int]) -> Tuple[int, int]:
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
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        return (img_size[1] - text_size[0] - 50, text_size[1] + 50)

    def get_top_left_screen_text_position(self, text: str, font: int, font_scale: float, font_thickness: int, img_size: Tuple[int, int]) -> Tuple[int, int]:
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
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        x_offset = img_size[0] // 25
        y_offset = img_size[1] // 45
        return (x_offset, text_size[1] + y_offset)

    def get_center_screen_text_position(self, text: str, font: int, font_scale: float, font_thickness: int, img_size: Tuple[int, int]) -> Tuple[int, int]:
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
        # Get the width and height of the text
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        # Calculate the position of the text on the screen, centered horizontally and vertically
        x = (img_size[1] // 2) - (text_size[0] // 2)  # Center horizontally
        y = (img_size[0] // 2) + (text_size[1] // 2)  # Center vertically with baseline adjustment
        return (x, y)

    def configure_text_settings(self, text_type: str, img_size: Tuple[int, int], text: Union[str, List[str]]) -> Tuple:
        """
        Configures text settings based on the type of text (counter or message).

        Args:
            text_type (str): The type of text ('counter' or 'message').
            image (np.ndarray): The image to draw on.
            text (str): The text to draw.

        Returns:
            Tuple containing font, font_scale, thickness, position, and border_thickness.
        """
        if text_type == "counter":
            font = getattr(cv2, counter_config["font"])
            font_scale = counter_config["font_scale"]
            thickness = font_scale * 2
            position = self.get_top_right_screen_text_position(text, font, font_scale, thickness, img_size)
        elif text_type == "double_counter":
            font = getattr(cv2, double_counter_config["font"])
            font_scale = double_counter_config["font_scale"]
            thickness = font_scale * 2
            positionl = self.get_top_right_screen_text_position(text[0], font, font_scale, thickness, img_size)
            positionr = self.get_top_right_screen_text_position(text[1], font, font_scale, thickness, img_size)
            text_height = cv2.getTextSize("0", font, font_scale, thickness)[0][1]
            r_y_offset = text_height + text_height // 2
            y_neg_offset = - text_height // 2
            positionr = (positionr[0], positionr[1] + y_neg_offset + r_y_offset)
            positionl = (positionl[0], positionl[1] + y_neg_offset)
            border_thickness = thickness + 2
            return (font, font_scale, thickness, [positionl, positionr], border_thickness)
        elif text_type == "message":
            font = getattr(cv2, message_config["font"])
            font_scale = message_config["font_scale"]
            thickness = font_scale * 2
            position = self.get_bottom_center_screen_text_position(text, font, font_scale, thickness, img_size)
        elif text_type == "title":
            font = getattr(cv2, title_config["font"])
            font_scale = title_config["font_scale"]
            thickness = font_scale * 2
            position = self.get_top_center_screen_text_position(text, font, font_scale, thickness, img_size)
        elif text_type == "sets":
            font = getattr(cv2, sets_config["font"])
            font_scale = sets_config["font_scale"]
            thickness = font_scale * 2
            position = self.get_top_left_screen_text_position(text, font, font_scale, thickness, img_size)
        elif text_type == "timer":
            font = getattr(cv2, timer_config["font"])
            font_scale = timer_config["font_scale"]
            thickness = font_scale * 2
            position = self.get_top_left_screen_text_position(text, font, font_scale, thickness, img_size)
            y_offset = img_size[1] // 20
            position = (position[0], position[1] + y_offset)
        elif text_type == "stats":
            font = getattr(cv2, stats_config["font"])
            font_scale = stats_config["font_scale"]
            thickness = font_scale * 2
            position = self.get_center_screen_text_position(text, font, font_scale, thickness, img_size)

        border_thickness = thickness + 2
        return font, font_scale, thickness, position, border_thickness

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

    def draw_start_pose(self, start_pose_image: np.ndarray, image: np.ndarray, opacity: float) -> np.ndarray:
        """
        Draws the start pose on an image with a given opacity.

        Args:
            start_pose_image (np.ndarray): The start pose image with an alpha channel.
            image (np.ndarray): Image to draw on.
            opacity (float): Opacity of the start pose image.
        """
        # Check if the image has an alpha channel
        if start_pose_image.shape[2] == 4:
            # Extract the alpha channel as a mask and apply opacity
            alpha_channel = (start_pose_image[:,:,3] / 255.0) * opacity
        else:
            raise ValueError("Start pose image does not have an alpha channel.")

        # Convert to BGR
        start_pose_image = cv2.cvtColor(start_pose_image, cv2.COLOR_BGRA2BGR)

        # Calculate the ratio of the new height to the old height
        ratio = image.shape[0] / start_pose_image.shape[0]

        # Calculate new width and height
        new_height = image.shape[0]
        new_width = int(start_pose_image.shape[1] * ratio)

        # Resize the image and the alpha channel, preserving the aspect ratio
        start_pose_image = cv2.resize(start_pose_image, (new_width, new_height))
        alpha_channel_resized = cv2.resize(alpha_channel, (new_width, new_height))

        # Calculate top-left corner position to center the image on the background
        x_offset = (image.shape[1] - new_width) // 2
        y_offset = (image.shape[0] - new_height) // 2

        # Blend the start pose image with the background image
        for c in range(0, 3):
            image[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c] = \
                alpha_channel_resized * start_pose_image[:, :, c] + \
                (1 - alpha_channel_resized) * image[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c]

        return image

    def draw_menu_images(self, options: List, image: np.ndarray, selected_option: int = None) -> np.ndarray:
        """
        Draws the menu images on a given frame.

        Args:
            options (list): A list of dictionaries containing the menu options.
            image (np.ndarray): The frame (image) to draw the menu images on.
            selected_option (int): The index of the selected option.

        Returns:
            np.ndarray: The updated frame with menu images drawn.
        """
        for i, option in enumerate(options):
            if selected_option is not None and i == selected_option:
                menu_image = option['image_selected']
            else:
                menu_image = option['image']

            position = option['position']
            x, y = int(position[0]), int(position[1])

            if menu_image.shape[2] == 4:
                # Extract the alpha channel as a mask and normalize it to [0, 1]
                alpha_channel = menu_image[:, :, 3] / 255.0
                alpha_channel_resized = cv2.resize(alpha_channel, (menu_image.shape[1], menu_image.shape[0]), interpolation=cv2.INTER_AREA)

                # Convert to BGR
                menu_image_bgr = cv2.cvtColor(menu_image, cv2.COLOR_BGRA2BGR)

                # Resize the image to match the frame size at the position
                h, w = menu_image_bgr.shape[:2]
                image_section = image[y:y+h, x:x+w]

                # Perform alpha blending
                for c in range(3):
                    image_section[:, :, c] = (alpha_channel_resized * menu_image_bgr[:, :, c] +
                                            (1 - alpha_channel_resized) * image_section[:, :, c])

                # Place the blended section back onto the image
                image[y:y+h, x:x+w] = image_section

        return image
    
    # def draw_numeric_menu(self, options: List, image: np.ndarray, selected_option: int = None) -> np.ndarray:
    #     """
    #     Draws the numeric menu on a given frame.

    #     Args:
    #         options (list): A list of dictionaries containing the menu options.
    #         image (np.ndarray): The frame (image) to draw the numeric menu on.
    #         selected_option (int): The index of the selected option.

    #     Returns:
    #         np.ndarray: The updated frame with numeric menu drawn.
    #     """
    #     for option in options:
    #         i = option['index'] - 1
    #         text = option['text']
    #         position = option['position']
    #         fg_image = option['text_selected_foreground']
    #         fg_position = option.get('text_selected_foreground_position')
    #         font = getattr(cv2, numeric_menu_config["font"])
    #         font_scale = numeric_menu_config["font_scale"]
    #         thickness = font_scale * 2
    #         color = (255, 255, 255)  # Default text color
    #         border_thickness = thickness + 2

    #         # Draw the text
    #         self.draw_text_with_border(image, text, position, font, font_scale, thickness, color, border_thickness)

    #         # Check if this option is the selected one
    #         if i == selected_option:
    #             # Get the position of the foreground image
    #             fg_x, fg_y = int(fg_position[0]), int(fg_position[1])
    #             # Get the foreground image size
    #             h, w = fg_image.shape[:2]
    #             # Ensure the foreground image is correctly positioned and blended
    #             image_section = image[fg_y:fg_y+h, fg_x:fg_x+w]
    #             alpha_channel = fg_image[:, :, 3] / 255.0
    #             for c in range(3):
    #                 image_section[:, :, c] = (alpha_channel * fg_image[:, :, c] +
    #                                         (1 - alpha_channel) * image_section[:, :, c])
    #             image[fg_y:fg_y+h, fg_x:fg_x+w] = image_section

    #     return image

    def draw_numeric_menu(self, options: List, image: np.ndarray, selected_option: int = None) -> np.ndarray:
        """
        Draws the numeric menu on a given frame.

        Args:
            options (list): A list of dictionaries containing the menu options.
            image (np.ndarray): The frame (image) to draw the numeric menu on.
            selected_option (int): The index of the selected option.

        Returns:
            np.ndarray: The updated frame with numeric menu drawn.
        """
        for option in options:
            text = option['text']
            position = option['position']
            font = getattr(cv2, numeric_menu_config['font'])
            font_scale = numeric_menu_config['font_scale']
            thickness = font_scale * 2
            color = (255, 255, 255)
            border_thickness = thickness + 2

            # Draw the text
            self.draw_text_with_border(image, text, position, font, font_scale, thickness, color, border_thickness)

            if option['index'] == selected_option:
                fg_position = option['text_selected_foreground_position']
                fg_x, fg_y = int(fg_position[0]), int(fg_position[1])
                fg_image = option['text_selected_foreground']
                h, w = fg_image.shape[:2]

                # Adjust position and clip the size to fit within the image
                fg_x = max(0, min(fg_x, image.shape[1] - w))
                fg_y = max(0, min(fg_y, image.shape[0] - h))
                h = min(h, image.shape[0] - fg_y)
                w = min(w, image.shape[1] - fg_x)

                # Ensure the foreground image is correctly positioned and blended
                image_section = image[fg_y:fg_y+h, fg_x:fg_x+w]
                alpha_channel = fg_image[:, :, 3] / 255.0
                for c in range(3):
                    image_section[:, :, c] = (alpha_channel * fg_image[:, :, c] +
                                            (1 - alpha_channel) * image_section[:, :, c])
                image[fg_y:fg_y+h, fg_x:fg_x+w] = image_section

        return image

    def draw_text_with_border(self, image: np.ndarray, text: str, position: Tuple[int, int], font: int, font_scale: int, thickness: int, color: Tuple[int, int, int], border_thickness: int) -> None:
        """
        Draws text with a border on an image.

        Args:
            image (np.ndarray): The image to draw on.
            text (str): The text to draw.
            position (Tuple[int, int]): The position to draw the text.
            font (int): The font type.
            font_scale (int): The scale of the font.
            thickness (int): The thickness of the font.
            color (Tuple[int, int, int]): The color of the text.
            border_thickness (int): The thickness of the border.
        """
        # Draw black border by offsetting text
        for x_offset in [-1, 1]:
            for y_offset in [-1, 1]:
                border_position = (position[0] + x_offset, position[1] + y_offset)
                cv2.putText(image, text, border_position, font, font_scale, (0, 0, 0), border_thickness, cv2.LINE_AA)

        # Draw the main text
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    def draw_output_on_image(self, output: Dict[str, str], image: np.ndarray) -> None:
        """
        Draws the output on an image.

        Args:
            output (Dict[str, str]): Output to draw.
            image (np.ndarray): Image to draw on.
        """
        for text_type in ['title', 'counter', 'sets', 'message', 'timer']:
            if text := output.get(text_type, ""):
                if isinstance(text, str):
                    font, font_scale, thickness, position, border_thickness = self.configure_text_settings(text_type, image.shape, text)
                    self.draw_text_with_border(image, text, position, font, font_scale, thickness, (255, 255, 255), border_thickness)
                elif isinstance(text, list):
                    text = [f"L {text[0]}/{output.get('reps_per_set', '')}", f"R {text[1]}/{output.get('reps_per_set', '')}"]
                    font, font_scale, thickness, position, border_thickness = self.configure_text_settings('double_counter', image.shape, text)
                    self.draw_text_with_border(image, text[0], position[0], font, font_scale, thickness, (255, 255, 255), border_thickness)
                    self.draw_text_with_border(image, text[1], position[1], font, font_scale, thickness, (255, 255, 255), border_thickness)

    def draw_title_background_banner(self, image: np.ndarray, title_banner: np.ndarray, transparency=0.5) -> None:
        """
        Draws the banner for the title and counters.
        """
        # Extract the alpha channel as a mask and adjust with transparency factor
        alpha_channel = (title_banner[:, :, 3] / 255.0) * transparency  # Normalize and apply transparency

        # Convert title_banner to BGR (discard alpha channel)
        title_banner_bgr = cv2.cvtColor(title_banner, cv2.COLOR_BGRA2BGR)

        # Assuming the banner will be placed at the top of the image
        y1, y2 = 0, title_banner.shape[0]
        x1, x2 = 0, title_banner.shape[1]

        # Extract the region of interest (ROI) from the original image
        roi = image[y1:y2, x1:x2]

        # Blend the banner and the ROI based on the alpha mask
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + title_banner_bgr[:, :, c] * alpha_channel

        # Place the blended ROI back into the original image
        image[y1:y2, x1:x2] = roi

    def draw_message_background_banner(self, image: np.ndarray, message_banner: np.ndarray, transparency=0.5) -> None:
        """
        Draws the banner for the message.
        """
        # Extract the alpha channel as a mask and adjust with transparency factor
        alpha_channel = (message_banner[:, :, 3] / 255.0) * transparency

        # Convert message_banner to BGR (discard alpha channel)
        message_banner_bgr = cv2.cvtColor(message_banner, cv2.COLOR_BGRA2BGR)

        # Assuming the banner will be placed at the bottom of the image
        y1, y2 = image.shape[0] - message_banner.shape[0], image.shape[0]
        x1, x2 = 0, message_banner.shape[1]

        # Extract the region of interest (ROI) from the original image
        roi = image[y1:y2, x1:x2]

        # Blend the banner and the ROI based on the alpha mask
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + message_banner_bgr[:, :, c] * alpha_channel

        # Place the blended ROI back into the original image
        image[y1:y2, x1:x2] = roi

    def draw_stats(self, stats: Dict[str, str], image: np.ndarray, stats_banner: np.ndarray, transparency=0.5) -> None:
        """
        Draws the stats background banner on the center of the image then draws the stats on the banner.

        Args:
            stats (Dict[str, str]): Stats to draw. Contains title, total_reps, and total_time.
            image (np.ndarray): Image to draw on.
        """
        # Extract the alpha channel as a mask and adjust with transparency factor
        alpha_channel = (stats_banner[:, :, 3] / 255.0) * transparency

        # Convert stats_banner to BGR (discard alpha channel)
        stats_banner_bgr = cv2.cvtColor(stats_banner, cv2.COLOR_BGRA2BGR)

        # Assuming the banner will be placed at the center of the image
        y1, y2 = image.shape[0] // 2 - stats_banner.shape[0] // 2, image.shape[0] // 2 + stats_banner.shape[0] // 2
        x1, x2 = image.shape[1] // 2 - stats_banner.shape[1] // 2, image.shape[1] // 2 + stats_banner.shape[1] // 2

        # Extract the region of interest (ROI) from the original image
        roi = image[y1:y2, x1:x2]

        # Blend the banner and the ROI based on the alpha mask
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + stats_banner_bgr[:, :, c] * alpha_channel

        # Place the blended ROI back into the original image
        image[y1:y2, x1:x2] = roi
        
        # Draw stats title, total reps, and total time on separate lines on the banner
        font, font_scale, thickness, position, border_thickness = self.configure_text_settings('stats', image.shape, stats['title'])
        y_offset_new_line = cv2.getTextSize("0", font, font_scale, thickness)[0][1] + 30
        position = (position[0], position[1] - int(y_offset_new_line * 1.5))
        self.draw_text_with_border(image, stats['title'], position, font, font_scale, thickness, (255, 255, 255), border_thickness)

        font, font_scale, thickness, position, border_thickness = self.configure_text_settings('stats', image.shape, stats['total_reps'])
        position = (position[0], position[1] + int(y_offset_new_line * 0.4))
        self.draw_text_with_border(image, stats['total_reps'], position, font, font_scale, thickness, (255, 255, 255), border_thickness)

        font, font_scale, thickness, position, border_thickness = self.configure_text_settings('stats', image.shape, stats['total_time'])
        position = (position[0], position[1] + int(y_offset_new_line * 1.4))
        self.draw_text_with_border(image, stats['total_time'], position, font, font_scale, thickness, (255, 255, 255), border_thickness)

    @staticmethod
    def create_rounded_banner(width: int, height: int, corner_radius=50) -> np.ndarray:
        """
        Creates a rounded banner.
        """
        # Background color
        background_color = stats_background_color

        # Create an empty image with an alpha channel (RGBA)
        banner = np.zeros((height, width, 4), dtype=np.uint8)

        # Draw the rectangle and circles in BGR color
        color = (*background_color, 255)  # Full opacity for the color part

        # Drawing filled circles at corners for rounded effect
        cv2.circle(banner, (corner_radius, corner_radius), corner_radius, color, -1)
        cv2.circle(banner, (width - corner_radius, corner_radius), corner_radius, color, -1)
        cv2.circle(banner, (corner_radius, height - corner_radius), corner_radius, color, -1)
        cv2.circle(banner, (width - corner_radius, height - corner_radius), corner_radius, color, -1)

        # Drawing filled rectangles to fill the rest of the banner
        cv2.rectangle(banner, (corner_radius, 0), (width - corner_radius, height), color, -1)
        cv2.rectangle(banner, (0, corner_radius), (width, height - corner_radius), color, -1)

        return banner

    @staticmethod
    def draw_selected_halo_from_alpha_channel(image: np.ndarray, halo_color: Tuple[int, int, int], halo_thickness: int) -> np.ndarray:
        """
        Draws a halo effect on an image using the alpha channel.

        Args:
            image (np.ndarray): Image to draw on.
            halo_color (Tuple[int, int, int]): Color of the halo.
            halo_thickness (int): Thickness of the halo.

        Returns:
            np.ndarray: The updated image with the halo effect drawn.
        """
        if image.shape[2] != 4:
            raise ValueError("Input image must have an alpha channel.")

        # Extract alpha channel
        original_alpha = image[:, :, 3].astype(float)

        # Ensure halo_thickness is odd and greater than 0
        halo_thickness = max(1, halo_thickness)
        blur_size = halo_thickness * 2
        blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size

        # Create a halo mask using the alpha channel
        halo_mask = cv2.GaussianBlur(original_alpha, (blur_size, blur_size), 0)

        # Create a halo image (RGB only)
        halo = np.zeros_like(image[:, :, :3])
        halo[:] = halo_color  # Fill with halo color

        # Ensure the mask is in the range of [0, 1]
        halo_mask = halo_mask / 255.0

        # Apply the halo mask to the halo image
        halo = halo.astype(float)
        for c in range(3):
            halo[:, :, c] = halo[:, :, c] * halo_mask

        # Combine the halo with the original image
        image_rgb = image[:, :, :3].astype(float)
        combined_rgb = np.where(halo_mask[:, :, None] > 0, np.clip(image_rgb + halo, 0, 255), image_rgb)

        # Create a new alpha channel where the halo is added to the original alpha
        new_alpha = np.clip(original_alpha + (halo_mask * 255), 0, 255).astype(np.uint8)

        return np.dstack((combined_rgb.astype(np.uint8), new_alpha))

    @staticmethod
    def play_sound(file_path):
        """
        Plays a sound file.

        Args:
            file_path (str): The path to the sound file.
        """
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        wave_obj.play()
