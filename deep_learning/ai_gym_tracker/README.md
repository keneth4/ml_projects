# AI Gym Tracker

AI Gym Tracker is an innovative application that combines the power of the MediaPipe Pose Landmarker model with advanced hand gesture interactions to provide a unique fitness tracking experience. This project leverages machine learning to identify key body locations, analyze posture, and categorize movements, making it an invaluable tool for fitness enthusiasts and professionals alike.

Key characteristics of AI Gym Tracker include:
- **Full Hand Gesture Menu**: Intuitive control through sophisticated hand gesture recognition, enabling seamless navigation and interaction.
- **Easy Configuration**: Utilize a YAML file for straightforward and flexible configuration, ensuring a user-friendly setup process.
- **Robust and Scalable Structure**: The project is built with a scalable architecture in mind, making it robust and suitable for future enhancements and integrations.
- **Easy Deployment**: Designed for straightforward deployment, allowing users to get the system up and running with minimal hassle.


## Model Description

### Pose Landmarker Model

The MediaPipe Pose Landmarker task is at the heart of this project. It detects landmarks of human bodies in an image or video. The model outputs body pose landmarks in both image coordinates and 3-dimensional world coordinates.

The pose landmarker model tracks 33 body landmark locations, representing various body parts, as illustrated below:

![Pose Landmarks](https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png)

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
5. Navigate to the root folder of the project in your command line or terminal.
```
    cd ai-gym-tracker
```
6. Run the application using the command:
```
    python -m src.app.main
```

## Contributing

We welcome contributions to AI Gym Tracker! If you have suggestions or improvements, feel free to fork this repository and submit a pull request.