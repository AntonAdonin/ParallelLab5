import itertools
import logging
import multiprocessing
import sys
import time
from multiprocessing import Pool

import cv2
import ffmpeg
from ultralytics import YOLO

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s",
                    handlers=[
                        logging.FileHandler("file.log", encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)


class VideoInfoReader:
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def __del__(self):
        self.cap.release()


class VideoFrameReader:
    def __init__(self, filename, left_frame, right_frame):
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.cap = cv2.VideoCapture(filename)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, left_frame)

    def __iter__(self):
        while self.cap.isOpened() and self.left_frame < self.right_frame:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            ret, frame = self.cap.read()
            self.left_frame += 1
            if not ret:
                logger.error("Can't receive frame (stream end?). Exiting ...")
                break
            yield ret, frame

    def __del__(self):
        self.cap.release()


class VideoFrameWriter:
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


def process_video_multiprocessing(filename, group_number, total_groups):
    print(filename, group_number)
    model = YOLO('yolov8s-pose.pt')
    cap = cv2.VideoCapture(filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_process = total_frames // total_groups
    left = frames_per_process * group_number
    right = left + frames_per_process - 1 if group_number != total_groups - 1 else total_frames
    print(f"start {group_number}: {frames_per_process, left, right}")

    results = []
    reader = VideoFrameReader(filename, left, right)
    for ret, frame in reader:
        if not ret:
            break
        result = model(frame, verbose=False)
        results.append(result[0].plot())
    # Высвобождаем ресурсы
    print(f"end {group_number}: {reader.left_frame} {reader.right_frame}")

    return results


if __name__ == '__main__':
    filename_input = "dima.MOV"
    filename_output = "result.mov"
    cpu_num = multiprocessing.cpu_count()
    print("Asd")
    t = time.time()
    with Pool(processes=cpu_num) as pool:
        args = zip(itertools.repeat(filename_input), range(cpu_num), itertools.repeat(cpu_num))
        result = pool.starmap(process_video_multiprocessing, args)
    t1 = time.time()
    logger.info(f"Frame processed in {t1 - t} s.")
    t = time.time()

    info = VideoInfoReader(filename_input)

    writer = VideoFrameWriter("tmp.mp4", info.fps, info.width, info.height)
    for group in result:
        for frame in group:
            writer.write(frame)
    t1 = time.time()
    logger.info(f"File wrote in {t1 - t} s.")

    input_video = ffmpeg.input("tmp.mp4")
    input_audio = ffmpeg.input(filename_input)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(filename_output).run()
