# ==========================================================================
# Intelligent Posture Assessment Using Machine Learning - VERSION 1
# (extracted from page 16 of the PDF)
# Basic side-view posture check using MediaPipe Pose
# ==========================================================================
#
# NOTE: The PDF listed this as a line to run in a terminal (not inside the
# script itself, since "pip install ..." is not valid Python syntax):
#
#   pip install opencv-python mediapipe numpy
#
# Run that once in your terminal before running this file.
# ==========================================================================

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def calculate_angle(point1, point2, point3):
    a = np.array(point1)  # First
    b = np.array(point2)  # Mid
    c = np.array(point3)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def is_side_view(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Simple check: Shoulders and hips should be vertically aligned
    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
    hip_diff = abs(left_hip.y - right_hip.y)

    if shoulder_diff < 0.1 and hip_diff < 0.1:
        return True
    return False


cap = cv2.VideoCapture(0)
threshold_angle = 30  # Example threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Check if the camera is aligned for the side view
        if is_side_view(landmarks):
            cv2.putText(frame, 'Side View: Aligned', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Calculate neck and torso inclination
            neck_angle = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            )

            torso_angle = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            )

            cv2.putText(frame, f'Neck Angle: {int(neck_angle)}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Torso Angle: {int(torso_angle)}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Check if the person is bending below the threshold angle
            if neck_angle > threshold_angle:
                cv2.putText(frame, 'Neck Bend Detected', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if torso_angle > threshold_angle:
                cv2.putText(frame, 'Torso Bend Detected', (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Side View: Not Aligned', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
