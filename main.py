import argparse
import copy
import itertools
import logging
import multiprocessing
import os
import shutil
import sys
import time
from multiprocessing.pool import Pool
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

    def __repr__(self):
        return f"fps: {self.fps}, total: {self.total_frames}, width: {self.width}, height: {self.height}"


class VideoFrameReader:
    def __init__(self, filename, left_frame=0, right_frame=None):
        self.left_frame = left_frame
        self.cap = cv2.VideoCapture(filename)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, left_frame)
        if right_frame is None:
            self.right_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        else:
            self.right_frame = right_frame

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

    def close(self):
        self.out.release()

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        logger.info("Finished writing frames in %.2f seconds", time.time() - self.start_time)


def process_video_multiprocessing(filename, group_number, total_groups):
    model = YOLO('yolov8s-pose.pt')
    info_reader = VideoInfoReader(filename)
    total_frames = info_reader.total_frames
    frames_per_process = total_frames // total_groups
    left = frames_per_process * group_number
    right = left + frames_per_process - 1 if group_number != total_groups - 1 else total_frames
    results = []
    reader = VideoFrameReader(filename, left, right)
    for ret, frame in reader:
        if not ret:
            break
        result = model(frame, verbose=False)
        results.append(result[0].plot())
    return results


def process_video_serial(filename):
    model = YOLO('yolov8s-pose.pt')
    info_reader = VideoInfoReader(filename)
    total_frames = info_reader.total_frames
    results = []
    reader = VideoFrameReader(filename, 0, total_frames)
    for ret, frame in reader:
        if not ret:
            break
        result = model(frame, verbose=False)
        results.append(result[0].plot())
    return results


def resize_video_multiprocessing(filename, group_number, total_groups):
    info_reader = VideoInfoReader(filename)
    total_frames = info_reader.total_frames
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
    parser = argparse.ArgumentParser(
        prog='Yolov8 pose',
        description='Program process video with yolov8s-pose in serial or parallel mode',
        epilog='All parameters are optional')
    parser.add_argument('-f', '--file', help="??camera name??", default="input.mp4")  # positional argument
    parser.add_argument('-m', '--mode',
                        help="singe or multi. defines serial or parallel mode",
                        default="multi")  # option that takes a value
    parser.add_argument('-o', '--output', help="output file for result", default='output.mp4')
    parser.add_argument('-r', '--resize', help="auto resize wideo to 640x480", default=True, type=bool)
    args = parser.parse_args()
    logger.debug(f"--file {args.file} --mode {args.mode} --output {args.output} --resize {args.resize}")
    cpu_num = multiprocessing.cpu_count()
    logger.info(f"{cpu_num} CPU cores available")
    logger.info(f"Running in {args.mode.upper()}processing mode")

    processes_results = []
    tmp_dir = "./tmp"
    logger.info("Trying to clear tmp directory")
    try:
        shutil.rmtree(tmp_dir)
        logger.info("Tmp directory flushed successfully")
    except Exception as error:
        logger.exception(f"deleting tmp directory: {error}")
    os.makedirs(tmp_dir, exist_ok=True)
    input = args.file
    output = args.output
    input_copy = copy.copy(input)
    input_filename = str(Path(input).name)

    info = VideoInfoReader(input)
    if args.resize:
        t = time.time()
        with Pool(processes=cpu_num) as pool:
            func_args = zip(itertools.repeat(input), range(cpu_num), itertools.repeat(cpu_num))
            processes_results = pool.starmap(resize_video_multiprocessing, func_args)
        t1 = time.time()
        logger.info(f"Video resized in {t1 - t} s.")
        input_copy = f"{tmp_dir}/RESIZED_{input_filename}"
        with VideoFrameWriter(input_copy, int(info.fps), 640, 480) as writer:
            for group in processes_results:
                for frame in group:
                    writer.write(frame)

    t = time.time()
    if args.mode == "multi":
        with Pool(processes=cpu_num) as pool:
            processes_results = pool.starmap(process_video_multiprocessing,
                                             zip(itertools.repeat(input_copy), range(cpu_num),
                                                 itertools.repeat(cpu_num)))
    else:
        process_video_serial(input_copy)
    t1 = time.time()
    logger.info(f"Video processed in {t1 - t} s.")

    info = VideoInfoReader(input_copy)
    input_copy = f"{tmp_dir}/PREDICTED_{input_filename}"
    with VideoFrameWriter(input_copy, int(info.fps), info.width, info.height) as writer:
        for group in processes_results:
            for frame in group:
                writer.write(frame)
    input_video = ffmpeg.input(input_copy)
    input_audio = ffmpeg.input(input)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output, loglevel="quiet").overwrite_output().run()
