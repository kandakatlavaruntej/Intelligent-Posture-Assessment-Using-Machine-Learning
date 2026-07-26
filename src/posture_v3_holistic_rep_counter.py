# ==========================================================================
# Intelligent Posture Assessment Using Machine Learning - VERSION 3
# (extracted from pages 18-19 of the PDF - both pages contained identical
#  code, so this single file covers both)
#
# MediaPipe Holistic: right-arm rep (curl) counter + neck/torso posture
# check, plus drawing of face mesh, both hands, and full body pose.
# ==========================================================================
#
# NOTE: install dependencies once in your terminal before running this file:
#   pip install opencv-python mediapipe numpy
# ==========================================================================

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Holistic and Drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Variables to keep track of rep count
counter = 0
stage = None

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():  # Check if webcam is open or not
        ret, frame = cap.read()  # Our image in the frame

        if not ret:
            break

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.setflags(write=0)  # Set writable to False

        # Make Detections
        results = holistic.process(image)  # Update results with new detections

        # Recolor image back to BGR for rendering
        image.setflags(write=1)  # Set writable to True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Recoloring back to RGB to BGR

        # Extract landmarks for the right arm
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == 'down':
                stage = "up"
                counter += 1
                print(counter)

            # Calculate neck and torso inclination
            neck_angle = calculate_angle(
                [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y],
                [landmarks[mp_holistic.PoseLandmark.NOSE.value].x,
                 landmarks[mp_holistic.PoseLandmark.NOSE.value].y],
                [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
            )

            torso_angle = calculate_angle(
                [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y],
                [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y],
                [landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].y]
            )

            cv2.putText(image, f'Neck Angle: {int(neck_angle)}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Torso Angle: {int(torso_angle)}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Check if the person is bending below the threshold angle
            threshold_angle = 30  # Example threshold
            if neck_angle > threshold_angle:
                cv2.putText(image, 'Neck Bend Detected', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if torso_angle > threshold_angle:
                cv2.putText(image, 'Torso Bend Detected', (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                   mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                   mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                   )

        # Draw right hand landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                   mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                   )

        # Draw left hand landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                   mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                   )

        # Draw pose detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                   )

        # Display rep count on the image
        cv2.putText(image, 'Reps: {}'.format(counter), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Raw Webcam Feed', image)  # Rendering our results to the screen

        if cv2.waitKey(10) & 0xFF == ord('q'):  # How we close our webcam
            break

cap.release()
cv2.destroyAllWindows()
