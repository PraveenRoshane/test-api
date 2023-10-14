import pickle
import json
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np


def get_highest_item_count(arr):
    """Get the most common item in an array."""
    return Counter(arr).most_common()


def process_video_and_predict(video_url):
    # Load pre-trained model
    with open("models/body_language.pkl", "rb") as f:
        model = pickle.load(f)

    # Process the video and make predictions
    cap = cv2.VideoCapture(video_url)
    with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        body_language_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            try:
                pose_landmarks = results.pose_landmarks.landmark
                face_landmarks = results.face_landmarks.landmark

                pose_data = list(
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            for landmark in pose_landmarks
                        ]
                    ).flatten()
                )
                face_data = list(
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            for landmark in face_landmarks
                        ]
                    ).flatten()
                )

                combined_data = pose_data + face_data

                prediction = model.predict([combined_data])[0]
                prediction_probabilities = model.predict_proba([combined_data])[0]

                body_language_results.append((prediction, prediction_probabilities))

            except AttributeError:
                pass

        cap.release()

    predicted_classes = [result[0] for result in body_language_results]
    highest_count = get_highest_item_count(predicted_classes)

    return json.dumps(highest_count)
