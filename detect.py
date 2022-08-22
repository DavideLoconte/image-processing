"""Perform detection and visualize result on window and command line"""

import numpy as np
import cv2

import model
import perspective
import measure

def detect_video(source, yolo, homography, distance, nowin, freq):
    """Detect violations in video sources"""
    number = 0
    for frame in source:
        number += 1
        predictions = model.predict(yolo, frame, 0.5)
        distances = measure.get_distances(frame, predictions, homography, distance)
        if freq == 0 or number % freq == 0:
            log(predictions, distances)
        visualize(frame, predictions, distances, homography)
        cv2.waitKey(1)

def detect_image(source, yolo, homography, distance, nowin):
    """Detect violation in images"""
    for frame in source:
        predictions = model.predict(yolo, frame, 0.5)
        distances = measure.get_distances(frame, predictions, homography, distance)
        log(predictions, distances)
        if not nowin:
            visualize(frame, predictions, distances, homography)
            cv2.waitKey(0)

def log(predictions, distances):
    """Log the results on stdout"""
    distances = np.sum(distances) / 2
    violations = 0
    print(f"Found {predictions.shape[0]} persons. Detected {violations} violations")

def visualize(frame, predictions, distances, homography = None):
    predict_frame = np.copy(frame)
    homography_frame = np.copy(frame)
    distance_frame = np.copy(frame)
    violations = measure.get_violations(distances)
    for i in range(len(violations)):
        color = (0,0,255) if 1 in violations[i] else (0,255,0)
        prediction = predictions[i].astype(int)
        pt1 = (prediction[0], prediction[1])
        pt2 = (prediction[2], prediction[3])
        predict_frame = cv2.rectangle(predict_frame, pt1, pt2, color, 3)

        for j in range(i+1, len(distances[i])):
            prediction_2 = predictions[j].astype(int)

            center_1 = (prediction[2] - prediction[0]) // 2 + prediction[0], (prediction[3])
            center_2 = (prediction_2[2] - prediction_2[0]) // 2 + prediction_2[0], (prediction_2[3])
            distance_frame = cv2.line(distance_frame, center_1, center_2, color=color, thickness=2)
            center = (center_1[0] + center_2[0]) // 2, (center_1[1] + center_2[1]) // 2
            distance_frame = cv2.putText(img=distance_frame, text=str(int(distances[i][j])) + " mm", org=center, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=color,thickness=1)

    if homography is not None:
        homography_frame = perspective.apply_homography(homography_frame, homography)

    final_frame = np.vstack([np.hstack([frame, homography_frame]), np.hstack([predict_frame, distance_frame])])
    cv2.imshow("Result", final_frame)
