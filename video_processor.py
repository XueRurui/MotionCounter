import datetime
import os
import cv2
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import A_00_Tools_PostureEncoding as pe
import A_01_Tools_Classification as pc
import A_02_Tools_NoiseReduction as rs
import A_03_Tools_Counter
import C_00_Show_Window as vs
from config import PATHS, CLASSIFIER_CONFIG, SMOOTHING_CONFIG


class VideoProcessor:
    def __init__(self, exercise_config):
        self.class_name = exercise_config['class_name']
        self.exercise_name = exercise_config['name']
        self.enter_threshold = exercise_config['enter_threshold']
        self.exit_threshold = exercise_config['exit_threshold']

        self.pose_tracker = mp_pose.Pose()
        self.pose_embedder = pe.FullBodyPoseEmbedder()
        self.pose_classifier = pc.PoseClassifier(
            pose_samples_folder=PATHS['output_csv'],
            pose_embedder=self.pose_embedder,
            class_name=self.class_name,
            **CLASSIFIER_CONFIG
        )
        self.pose_filter = rs.EMADictSmoothing(**SMOOTHING_CONFIG)
        self.counter = A_03_Tools_Counter.RepetitionCounter(
            class_name=self.class_name,
            enter_threshold=self.enter_threshold,
            exit_threshold=self.exit_threshold
        )
        self.visualizer = None

    def process_frame(self, input_frame):
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = self.pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(output_frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)

            frame_height, frame_width = output_frame.shape[:2]
            pose_landmarks = np.array([
                [lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                for lmk in pose_landmarks.landmark
            ], dtype=np.float32)

            pose_classification = self.pose_classifier(pose_landmarks)
            pose_classification_filtered = self.pose_filter(pose_classification)
            repetitions_count = self.counter(pose_classification_filtered)
        else:
            pose_classification = None
            pose_classification_filtered = self.pose_filter(dict())
            pose_classification_filtered = None
            repetitions_count = self.counter.n_repeats

        output_frame = self.visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count
        )
        return output_frame

    def close(self):
        self.pose_tracker.close()
