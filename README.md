# AI Hand Gesture Control System

> **A real-time Human-Computer Interaction (HCI) system that allows users to control the mouse and keyboard using hand gestures via a standard webcam.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Overview

This project replaces physical mouse and keyboard interactions with touchless hand gestures. It leverages **Deep Learning (Stacked LSTM)** to recognize dynamic gestures from video streams and maps them to OS-level commands using a robust **State Machine** logic.

The system is designed to be lightweight, running smoothly on CPU at **30 FPS**.

---

## Key Features

* **Real-time Hand Tracking:** Uses MediaPipe for high-speed skeletal tracking.
* **Dynamic Gesture Recognition:** Identifies complex actions (Click, Swipe, Zoom) using a custom-trained LSTM model.
* **Smart One-Shot Trigger:** Implements a **State Machine** to prevent command duplication (e.g., prevents accidental double-clicks when holding a gesture).
* **Anti-Jitter Smoothing:** Applies **Exponential Moving Average (EMA)** to ensure smooth and precise cursor movement, filtering out camera noise.
* **Robustness:** Uses **Wrist-Centric Normalization**, making recognition independent of user position in the frame.

---

## Tech Stack

* **Core:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **AI/Deep Learning:** TensorFlow (Keras), NumPy, Scikit-learn
* **Automation:** PyAutoGUI
* **Data Processing:** Pandas, Joblib

---

## Project Structure

```text
Hand-Gesture-Control/
│
├── Model_Output/           # Pre-trained models and scalers
│   ├── action_best_model.h5
│   ├── scaler.save
│   └── actions.json
│
├── convert_ipn.py          # Data preprocessing pipeline
├── train.py                # LSTM model training script
├── run_app.py              # Main application (Real-time inference)
├── requirements.txt        # Dependencies
└── README.md               # Documentation