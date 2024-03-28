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




def serial_video_process(input_filename, output_filename):
    model = YOLO('yolov8s-pose.pt')
    cap = cv2.VideoCapture(input_filename)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    return
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 1
    if not cap.isOpened():
        logger.error("Cannot open camera")
        exit()
    while cap.isOpened():
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
        if not ret:
            logger.error("Can't receive frame (stream end?). Exiting ...")
            break
        result = model(frame)
        logger.debug(f"frame number: {frame_number}/{frame_total}")
        out.write(result[0].plot())
        frame_number += 1
        # cv2.imshow("frame", result[0].plot())
    out.release()
    cap.release()
serial_video_process("full.mov", "res.mov")