# AI Gym Tracker

AI Gym Tracker is an innovative application that combines the power of the MediaPipe Pose Landmarker model with advanced hand gesture interactions to provide a unique fitness tracking experience. This project leverages machine learning to identify key body locations, analyze posture, and categorize movements, making it an invaluable tool for fitness enthusiasts and professionals alike.

Key characteristics of AI Gym Tracker include:
- **Full Hand Gesture Menu**: Intuitive control through sophisticated hand gesture recognition, enabling seamless navigation and interaction.
- **Easy Configuration**: Utilize a YAML file for straightforward and flexible configuration, ensuring a user-friendly setup process.
- **Robust and Scalable Structure**: The project is built with a scalable architecture in mind, making it robust and suitable for future enhancements and integrations.
- **Easy Deployment**: Designed for straightforward deployment, allowing users to get the system up and running with minimal hassle.

<center>
    <table>
    <tr>
        <th colspan="2" style="text-align: center;">Interactive Menu</th>
    </tr>
    <tr>
        <th style="text-align: center;">Exercise selection</th>
        <th style="text-align: center;">Reps per set selection</th>
    </tr>
    <tr>
        <td>
        <img src="https://github.com/keneth4/ml_projects/blob/main/deep_learning/ai_gym_tracker/src/assets/demo/exercise_menu_demo.gif?raw=true" width="300" alt="exercise menu demo">
        </td>
        <td>
        <img src="https://github.com/keneth4/ml_projects/blob/main/deep_learning/ai_gym_tracker/src/assets/demo/reps_menu_demo.gif?raw=true" width="300" alt="numeric menu demo">
        </td>
    </tr>
    <tr>
        <th style="text-align: center;">Angle measurement</th>
        <th style="text-align: center;">Both arms up detection</th>
    </tr>
    <tr>
        <td>
        <img src="https://github.com/keneth4/ml_projects/blob/main/deep_learning/ai_gym_tracker/src/assets/demo/acc_angle_demo.gif?raw=true" width="300" alt="angle meassurement demo">
        </td>
        <td>
        <img src="https://github.com/keneth4/ml_projects/blob/main/deep_learning/ai_gym_tracker/src/assets/demo/both_arms.gif?raw=true" width="300" alt="both arms up demo">
        </td>
    </tr>
    <tr>
        <th style="text-align: center;">Optimal starting pose check</th>
        <th style="text-align: center;">Squats Exercise</th>
    </tr>
    <tr>
        <td>
        <img src="https://github.com/keneth4/ml_projects/blob/main/deep_learning/ai_gym_tracker/src/assets/demo/starting_pose_demo.gif?raw=true" width="300" alt="starting pose demo">
        </td>
        <td>
        <img src="https://github.com/keneth4/ml_projects/blob/main/deep_learning/ai_gym_tracker/src/assets/demo/squats_demo.gif?raw=true" width="300" alt="squats demo">
        </td>
    </tr>
    <tr>
        <th colspan=2 style="text-align: center;">Session stats at finish</th>
    </tr>
    <tr>
        <td colspan=2 style="text-align: center;">
        <img src="https://github.com/keneth4/ml_projects/blob/main/deep_learning/ai_gym_tracker/src/assets/demo/stats_demo.gif?raw=true" width="300" alt="final stats demo">
        </td>
    </tr>
    </table>
</center>

For a full demonstration of AI Gym Tracker in action, check out our video demo:

<center>
  <a href="https://www.youtube.com/watch?v=Y8o3ex4k8zM">
    <img src="https://i3.ytimg.com/vi/Y8o3ex4k8zM/maxresdefault.jpg" width="500" alt="Video Demo">
  </a>
</center>

## Model Description

### Pose Landmarker Model

The MediaPipe Pose Landmarker task is at the heart of this project. It detects landmarks of human bodies in an image or video. The model outputs body pose landmarks in both image coordinates and 3-dimensional world coordinates.

The pose landmarker model tracks 33 body landmark locations, representing various body parts, as illustrated below:

<center>
<img src="https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png" width="500" alt="pose landmarks">
</center>

These landmarks include key points like the eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles.

### Exercise Rep Counter

One of the core functionalities of AI Gym Tracker is the ability to count exercise repetitions. By calculating the angle between three key points (for example, shoulder - elbow - wrist for a bicep curl exercise), the application can effectively track and count the repetitions of various exercises.

## Getting Started

To run AI Gym Tracker, follow these steps:

1. Clone or download this repository to your local machine.
```
    git clone https://github.com/keneth4/ml_projects.git
    cd ml_projects/deep_learning/ai_gym_tracker
```
2. Create a virtual environment.
    - On Windows:
    ```
        python -m venv venv
    ```
    - On macOS and Linux:
    ```
        python3 -m venv venv
    ```
3. Activate the virtual environment.
    - On Windows:
    ```
        .\venv\Scripts\activate
    ```
    - On macOS and Linux:
    ```
        source venv/bin/activate
    ```
4. Ensure that you have Python installed, along with the necessary dependencies (listed in `requirements.txt`).
```
    pip install -r requirements.txt
```
5. Run the application using the command:
```
    python -m src.app.main
```

## Contributing

We welcome contributions to AI Gym Tracker! If you have suggestions or improvements, feel free to fork this repository and submit a pull request.