# Model

## Machine Learning Algorithm

The posture assessment system uses a **Random Forest Classifier** trained on features extracted from human body pose landmarks.

## Feature Extraction

The system extracts anatomical landmarks using **MediaPipe Pose**, including:

- Neck alignment
- Shoulder alignment
- Spine posture
- Body inclination
- Sitting posture angles

These features are used to classify posture as either **Correct** or **Incorrect**.

## Libraries

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn

## Training

The Random Forest classifier was trained using manually labelled posture samples collected during the research.

## Model Availability

The original trained model (`.pkl` / `.joblib`) is not included in this repository because it is no longer available.

This repository focuses on documenting and demonstrating the implementation methodology presented in the associated research publication.
