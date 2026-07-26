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

---

# 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Computer Vision | OpenCV, MediaPipe |
| Machine Learning | Scikit-learn (Random Forest Classifier) |
| Numerical Computing | NumPy |
| Development Environment | Visual Studio Code, Jupyter Notebook |
| Input Device | Webcam |
| Operating System | Windows |

---

# 📂 Repository Structure

```text
intelligent-posture-assessment-using-machine-learning/
│
├── src/
│   ├── posture_v1_basic.py
│   ├── posture_v2_refactored.py
│   └── posture_v3_holistic_rep_counter.py
│
├── docs/
│   ├── dataset.md
│   ├── model.md
│   └── contribution.md
│
├── paper/
│   └── Research_Paper.pdf
│
├── report/
│   └── Project_Report.pdf
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/kandakatlavaruntej/intelligent-posture-assessment-using-machine-learning.git

cd intelligent-posture-assessment-using-machine-learning
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python src/posture_v3_holistic_rep_counter.py
```

> **Note:** The original research dataset and trained Random Forest model are not included in this repository. The repository focuses on the implementation methodology and real-time posture detection pipeline described in the accompanying publication.


---

# 📊 Results

The proposed posture assessment system achieved the following performance during evaluation on the custom dataset.

| Metric | Score |
|---------|-------|
| Accuracy | **95.3%** |
| Precision | **94.8%** |
| Recall | **96.1%** |
| F1-Score | **95.4%** |

## Highlights

- Real-time posture assessment using a standard webcam
- Lightweight implementation suitable for consumer hardware
- Accurate detection of correct and incorrect sitting posture
- Research-backed methodology validated through experimentation

  ---

# 📄 Research Publication

This project is based on our published research paper:

**Intelligent Posture Assessment Using Machine Learning**

**Journal:** Journal of Computational Analysis and Applications (JoCAAA)

**Volume:** 33

**Issue:** 6

**Year:** 2024

The complete publication is available in the **paper/** directory of this repository.

---

# 📚 Documentation

Additional documentation is available in the **docs/** directory.

- 📂 Dataset Documentation → `docs/dataset.md`
- 🤖 Model Documentation → `docs/model.md`
- 👨‍💻 Contribution Details → `docs/contribution.md`

---

# 🚀 Future Improvements

- Deep Learning based posture classification
- Multi-person posture assessment
- Mobile application support
- Cloud-based analytics dashboard
- Personalized posture recommendations
- Real-time posture history and reporting
- Cross-platform deployment

  ---

# 📜 License

This project is distributed under the **MIT License**.

See the `LICENSE` file for complete details.


