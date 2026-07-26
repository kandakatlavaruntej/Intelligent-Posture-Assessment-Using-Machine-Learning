# 🧠 Intelligent Posture Assessment Using Machine Learning

A real-time posture assessment system that uses **Computer Vision** and **Machine Learning** to detect incorrect sitting posture and provide immediate visual feedback.

Developed as part of an academic research project, this system combines **MediaPipe**, **OpenCV**, and a **Random Forest Classifier** to analyze human body landmarks and classify sitting posture with high accuracy.

---

## 📌 Overview

Poor sitting posture is one of the leading causes of musculoskeletal disorders among students and professionals. This project provides an intelligent, low-cost posture assessment solution that works with a standard webcam and performs real-time posture analysis without requiring wearable sensors.

The system extracts human pose landmarks using MediaPipe, computes posture-related features, and classifies posture as **Correct** or **Incorrect** using a trained Machine Learning model.

---

## ✨ Key Features

- Real-time posture detection
- MediaPipe-based human pose estimation
- OpenCV webcam integration
- Machine Learning posture classification
- Neck and spine alignment analysis
- Live visual posture feedback
- Lightweight and low-cost solution
- Research-backed implementation

  ---

# 🏗️ System Architecture

The posture assessment system follows a simple real-time processing pipeline:

```text
                  Webcam Input
                        │
                        ▼
              OpenCV Video Capture
                        │
                        ▼
         MediaPipe Pose Landmark Detection
                        │
                        ▼
        Feature Extraction (Pose Angles)
                        │
                        ▼
        Random Forest Classification
                        │
                        ▼
     Correct / Incorrect Posture Prediction
                        │
                        ▼
         Real-Time Visual Feedback
```

## Workflow

1. Capture live video frames using OpenCV.
2. Detect human body landmarks with MediaPipe Pose.
3. Extract posture-related geometric features.
4. Pass the extracted features to the trained Random Forest classifier.
5. Predict whether the sitting posture is **Correct** or **Incorrect**.
6. Display posture feedback on the video stream in real time.
