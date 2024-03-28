import logging
import sys

import cv2
from ultralytics import YOLO


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s",
                    handlers=[
                        logging.FileHandler("file.log", encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)

def serial_video_process(video_file, output_file):
    model = YOLO('yolov8s-pose.pt')
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        exit()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.error("Can't receive frame (stream end?). Exiting ...")
            break
        result = model(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        cv2.imshow("frame", result[0].plot())







serial_video_process("full.mov", "out_short.mov")