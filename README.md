# PoseEstimationAndGestureClassification
This repository contains MATLAB code for pose estimation and gesture classification using the PoseNet library. The project involves capturing images from a webcam, estimating human poses, and classifying gestures based on predefined landmarks.

## Overview
This project demonstrates the use of PoseNet for real-time pose estimation and gesture classification. The system captures images from a webcam, processes them to detect key points on the human body, and classifies the pose into predefined gestures.

## Features
- Real-time pose estimation using a webcam
- Classification of gestures into predefined categories
- Visualization of detected key points on the body

## Gestures
The system currently supports the following gestures:
1. Hands are down
2. Hands are up
3. Right hand is up
4. Left hand is up
5. Hands crossed (proposed)
6. Additional gestures can be added to the database

The gesture data is stored in a 3D matrix tab where each slice represents the coordinates of key points for a particular gesture.
