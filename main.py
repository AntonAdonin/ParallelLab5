import itertools
import logging
import multiprocessing
import sys
import time
from multiprocessing import Pool
import ffmpeg

import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s",
                    handlers=[
                        logging.FileHandler("file.log", encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)


class MyVideoReader:
    def __init__(self, filename, frame_n):
        self.cap = cv2.VideoCapture(filename)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n - 1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

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


def process_video_multiprocessing(filename, group_number, total_groups):
    print(filename, group_number)
    model = YOLO('yolov8s-pose.pt')
    cap = cv2.VideoCapture(filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_process = total_frames // total_groups
    print(total_frames)
    left = frames_per_process * group_number
    right = left + frames_per_process - 1 if group_number != total_groups - 1 else total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, left)
    print(f"start {group_number}: { frames_per_process, left, right}")
    # Получаем высоту, ширину и количество кадров в видео
    width, height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    results = []
    try:
        while left < right:
            ret, frame = cap.read()
            if not ret:
                break
            result = model(frame, verbose=False)
            results.append(result[0].plot())
            left += 1
    except:
        cap.release()
    # Высвобождаем ресурсы
    print(f"end {group_number}: {right}")


    cap.release()
    return results


if __name__ == '__main__':
    filename_input = "dima.MOV"
    filename_output = "result.mov"
    reader = MyVideoReader(filename_input, 0)
    fps = reader.fps
    w, h = reader.width, reader.height
    cpu_num = multiprocessing.cpu_count()
    t = time.time()
    with Pool(processes=cpu_num) as pool:
        args = zip(itertools.repeat(filename_input), range(cpu_num), itertools.repeat(cpu_num))
        result = pool.starmap(process_video_multiprocessing, args)
    t1 = time.time()
    logger.info(f"Frame processed in {t1 - t} s.")
    t = time.time()
    writer = MyVideoWriter("result.mov", fps, w, h)
    for group in result:
        for frame in group:
            writer.write(frame)
    t1 = time.time()
    logger.info(f"File wrote in {t1 - t} s.")


    input_video = ffmpeg.input(filename_output)
    input_audio = ffmpeg.input(filename_input)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output("ffmpeg_res.mov").run()