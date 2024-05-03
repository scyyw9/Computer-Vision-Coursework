import os
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
import deep_sort.deep_sort.deep_sort as ds


class VideoProcessor(object):
    def __init__(self, args, device):
        self.args = args
        # Set the target object categories to be tracked
        self.detect_class = 0  # 0 represents person
        self.model = YOLO(self.args.weights)  # Generate model and load pre-trained weights
        self.model.to(device)
        self.tracker = ds.DeepSort(device, self.args.tracker_weights)

    # Brightness adjustment
    @staticmethod
    def adjust_brightness(frame, value):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return frame

    # Saturation adjustment
    @staticmethod
    def adjust_saturation(frame, value):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, value)
        s[s > 255] = 255
        s[s < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return frame

    # Contrast enhancement
    @staticmethod
    def enhance_contrast(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        final_lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
        return frame

    @staticmethod
    def put_text_with_background(img, text, origin, font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1,
                                 text_color=(0, 0, 255), bg_color=(0, 0, 255), thickness=1):
        # Add ID prefix to the text
        id_text = f"ID:{text}"

        # Calculate the size of the text
        (text_width, text_height), _ = cv2.getTextSize(id_text, font, font_scale, thickness)

        # Draw the background
        bottom_left = origin
        top_right = (origin[0] + text_width, origin[1] - text_height - 5)  # Subtract 5 to leave some margin
        cv2.rectangle(img, bottom_left, top_right, bg_color, -1)

        # Draw the text on the rectangle
        text_origin = (origin[0], origin[1] - 5)
        cv2.putText(img, id_text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def extract_detections(results, detect_class):
        """
        Extract and process detection information from model results.
        - results: YoloV8 model prediction results, containing information such as the position, class,
          and confidence of the detected objects.
        - detect_class: the index of the target class to be extracted.
        """

        # Store the position information of the detected target
        detections = np.empty((0, 4))  # (bottom_left: x1, y1; top_right: x2, y2)

        confidence = []  # Store the confidence scores of the detected targets

        for r in results:
            for box in r.boxes:
                # If the detected target class matches the specified target class,
                # extract the position information and confidence score of the target
                if box.cls[0].int() == detect_class:
                    x1, y1, x2, y2 = box.xywh[0].int().tolist()  # Extract position information
                    conf = round(box.conf[0].item(), 2)  # Extract confidence score
                    detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                    confidence.append(conf)
        return detections, confidence

    def detect_and_track(self, input_path: str, output_path: str, detect_class: int, model, tracker):
        """
        Process the video, detect and track.
        - input_path: the path of the input video file.
        - output_path: the path to save the processed video.
        - detect_class: the index of the target class to be detected and tracked.
        - model: the model used for object detection.
        - tracker: the model used for object tracking.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video file {input_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the video FPS
        # Get the video width and height
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output_video_path = Path(output_path) / self.args.output

        # Set the video codec format to XVID format avi file to prevent video interruption
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size, isColor=True)

        # Read and process each frame of the video
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Use the model to perform object detection on the current frame
            original_frame = frame
            refined_frame = self.adjust_brightness(frame, 10)
            refined_frame = self.adjust_saturation(refined_frame, 5)
            refined_frame = self.enhance_contrast(refined_frame)

            results = model(refined_frame, stream=True)

            # Obtain detected information
            detections, confidence = self.extract_detections(results, detect_class)

            # Use the DeepSORT model to track the detected targets
            tracker_results = tracker.update(detections, confidence, refined_frame)

            for x1, y1, x2, y2, Id in tracker_results:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Draw the bounding boxes and text
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                self.put_text_with_background(original_frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1,
                                              text_color=(255, 255, 255), bg_color=(0, 0, 255))

            output_video.write(original_frame)

        output_video.release()
        cap.release()

        print(f'Results in: {output_video_path}')

    def process(self):
        print(self.args)
        print(f"Tracking {self.model.names[self.detect_class]}")
        self.detect_and_track(self.args.input_path, self.args.save_path, self.detect_class, self.model, self.tracker)
