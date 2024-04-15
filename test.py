import itertools
import logging
import multiprocessing
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

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
            yield ret, cv2.resize(frame, (640, 480))

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


def resize_video_multiprocessing(filename, group_number, total_groups):
    cap = cv2.VideoCapture(filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_process = total_frames // total_groups
    left = frames_per_process * group_number
    right = left + frames_per_process - 1 if group_number != total_groups - 1 else total_frames

    results = []
    reader = VideoFrameReader(filename, left, right)
    for ret, frame in reader:
        if not ret:
            break
        results.append(cv2.resize(frame, (640, 480)))

    return results


if __name__ == '__main__':
    cpu_num = multiprocessing.cpu_count()
    PROJECT_DIR = Path(__file__).parents[0]
    tmp_path = PROJECT_DIR / Path("./tmp")
    tmp_path.mkdir(exist_ok=True)
    print(cpu_num)
    input = "dima.mp4"
    input_copy = input
    output = "result.mov"

    print(tmp_path)
    info = VideoInfoReader(input)
    if info.width != 640 or info.height != 480:
        t = time.time()
        with Pool(processes=cpu_num) as pool:
            args = zip(itertools.repeat(input), range(cpu_num), itertools.repeat(cpu_num))
            result = pool.starmap(resize_video_multiprocessing, args)
        t1 = time.time()
        logger.info(f"Video resized in {t1 - t} s.")

        t = time.time()
        resized = Path(f"RESIZED_{input}")
        input_copy = str(tmp_path / resized)
        writer = VideoFrameWriter(input_copy, int(info.fps), info.width, info.height)
        for group in result:
            for frame in group:
                writer.write(frame)
        t1 = time.time()
        logger.info(f"Resized File wrote in {t1 - t} s.")

    t = time.time()
    with Pool(processes=cpu_num) as pool:
        args = zip(itertools.repeat(input_copy), range(cpu_num), itertools.repeat(cpu_num))
        result = pool.starmap(process_video_multiprocessing, args)
    t1 = time.time()
    logger.info(f"Video processed in {t1 - t} s.")

    t = time.time()
    file_path = tmp_path / Path(f"PREDICTED_{input}")
    writer = VideoFrameWriter(file_path, int(info.fps), info.width, info.height)
    for group in result:
        for frame in group:
            writer.write(frame)
    t1 = time.time()
    logger.info(f"Resized File wrote in {t1 - t} s.")

    input_video = ffmpeg.input(file_path)
    input_audio = ffmpeg.input(input)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output, loglevel="quiet").overwrite_output().run()
