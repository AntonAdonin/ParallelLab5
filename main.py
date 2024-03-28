import logging
import sys
from multiprocessing import Pool, TimeoutError
import time
import os

import cv2


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s",
                    handlers=[
                        logging.FileHandler("file.log", encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)

class MyVideoReader:
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        while self.cap.isOpened():
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Can't receive frame (stream end?). Exiting ...")
                break
            yield frame

    def __del__(self):
        self.cap.release()

class MyVideoWriter:
    def __init__(self, filename, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.width = width
        self.height = height
        self.fps = fps
        self.out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    def write(self, frame):
        self.out.write(frame)

    def __del__(self):
        self.out.release()


def process_frame(frame):
    # some intensive computation...
    frame = cv2.medianBlur(frame, 19)
    frame = cv2.medianBlur(frame, 19)
    return frame

if __name__ == '__main__':
    # start 4 worker processes
    reader = MyVideoReader("short.mov")
    # for i in reader:
    #     print(i)
    with Pool(processes=4) as pool:
        res = pool.map(process_frame, iter(reader))
        print(res)

    writer = MyVideoWriter("out.mov", reader.fps, reader.width, reader.height)
    for frame in res:
        writer.write(frame)
